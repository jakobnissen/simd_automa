using SIMD
import Automa: ByteSet, Machine, traverse
import Automa
using FASTX
const i128 = NTuple{16, VecElement{UInt8}}
const v128 = Vec{16, UInt8}

# Overall:
# Check if it's a self-edge with no actions
# If so, check both inverted byteset (I) and byteset (B).
# Pick the one with highest priority, and remember if it's inverted or not.
# Generate the constants, e.g. LUTS and such using the I or B and whether it's inverted.
# These LUTS should be in form of UInt128.
# Generate the code as function calls to @inline functions, and rely on constant
# propagation.

# Todo: Perhaps make it so that the vectors are constant UInt128s in the
# generated Julia code? Then it's portable and does not depend on silly
# consts

# Also, for each of these functions, I need a way to "invert" them.
# Right now, they produce all zeros if all bytes are within the byte set.
# I need them to produce all zeros if NO bytes are within the (inverted) byte set.
# So it must return 0x00 if it's not in the set.

# For allwithin/allwithin128, simply invert the content of the two LUTs
# 

# Generalize to 32-byte? Or is that too silly?
@inline function vpshufb(x::v128, mask::v128)
    v128(ccall("llvm.x86.ssse3.pshuf.b.128", llvmcall, i128, (i128, i128), x.data, mask.data))
end

const bitshift_mask = v128((0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,))

# This works similar to bitshifting 0x01 using the lower 3 bits of input
@inline bitshift_ones(shift::v128) = vpshufb(bitshift_mask, shift)
@inline tov128(x::UInt128) = unsafe_load(Ptr{v128}(pointer_from_objref(Ref(x))))
@inline haszerolayout(x::v128) = iszero(unsafe_load(Ptr{UInt128}(pointer_from_objref(Ref(x)))))
@inline haszerolayout(x::Vec{16, Bool}) = iszero(unsafe_load(Ptr{UInt128}(pointer_from_objref(Ref(x)))))
@inline load_v128(data::Vector{UInt8}, p::Int) = unsafe_load(Ptr{v128}(pointer(data, p)))
@inline load_v128(p::Ptr, i::Integer) = unsafe_load(Ptr{v128}(p) + i - 1)
# We have this to keep the same constant mask in memory.
@inline shrl4(x::v128) = x >>> 0x04

# Offset should be 0 for generic fallback. For within128, use offset and just
# take the lower. For ascii, dont use offset and take lower.
function generic_luts(byteset::ByteSet, offset::UInt8, invert::Bool)
    topzero = fill(invert ? 0x00 : 0xff, 16)
    topone = copy(topzero)
    for byte in byteset
        byte -= offset
        index = (byte & 0x0f) + 0x01
        bitmap = (byte & 0x80) == 0x80 ? topone : topzero
        shift = (byte >> 0x04) & 0x07
        bitmap[index] ⊻= 0x01 << shift
    end
    a = unsafe_load(Ptr{UInt128}(pointer(topzero)))
    b = unsafe_load(Ptr{UInt128}(pointer(topone)))
    return a, b
end

@inline function generic(x::v128, topzero::UInt128, topone::UInt128)
    lut_topzero, lut_topone = tov128(topzero), tov128(topone)
    y = x & 0b10001111
    lower = vpshufb(lut_topzero, y)
    upper = vpshufb(lut_topone, y ⊻ 0b10000000)
    bitmap = lower | upper
    return bitmap & bitshift_ones(shrl4(x))
end

# if top bit is set, bitmap will be 0xff, and no shift will be allowed, and it will guarantee
# an error.
# To construct map, use generic_luts[1] with the same offset. If ascii, set invert
@inline function within128(x::v128, ::Val{lut}, ::Val{offset}, ::Val{ascii}) where {lut, offset, ascii}
    f = ascii ? Base.:~ : identity
    bitmap = f(vpshufb(lut, x - offset))
    return bitmap & bitshift_ones(shrl4(x))
end

@inline function within128(x::v128, lutbytes::UInt128, ::Val{offset}, ::Val{ascii}) where {offset, ascii}
    lut = tov128(lutbytes)
    f = ascii ? Base.:~ : identity
    bitmap = f(vpshufb(lut, x - offset))
    return bitmap & bitshift_ones(shrl4(x))
end

# This is almost exactly the same as above - but the table is crafted
# differently. Here, it gives 8-bit bitarrays for all possible combinations.
# And there's no offset.
# E.g, if we name the 8 bytes A .. H, then we associate them in LUT2 with a bit
# 0x01,0x02,0x04...0x80. LUT1 then gives a byte each allowed bit is zerod.
# We then AND the LUT2 result with the LUT1 result, and result is zero if allowed.
@inline function eleme8(x::v128, ::Val{lut1}, ::Val{lut2})  where {lut1, lut2}
    # Get a 8-bit bitarray of the possible ones
    mask = vpshufb(lut1, x & 0b00001111)
    shifted = vpshufb(lut2, shrl4(x))
    return mask & shifted
end

@inline function inverted_range(x::v128, ::Val{notlow}, ::Val{nothigh}) where {notlow, nothigh}
    return (x > nothigh) | (x < notlow)
end

## Here's one where they're 16 apart at most.
@inline function allwithin16(x::v128, ::Val{lut}, ::Val{low}) where {lut, low}
    # topzero should simply return 0x00 for true or 0x01 for false
    lower = vpshufb(lut, (x - low) & 0b00001111)
    return lower | (x & 0b11110000) 
end

## Here's one where it's a single range:
@inline function withinrange(x::v128, ::Val{low}, ::Val{high}) where {low, high}
    return (x - low) > (high - low)
end

# Here's one with only one member - here 0xa3 as an example.
@inline allsame(x::v128, ::Val{member}) where member = x ⊻ member

#= Algorithm
Suppose, if we enumerate those with ending bits 0000, the allowed ones are 0, 1, 3, 5, 9, 11, 12, 13.
We can represent this with a bitmask, where 1 is disallowed:
15             0
v              v
1100010111010100
|  a   ||  b   |

we construct L1 and L2 such that
y = x & 0b10001111
a = vpshufb(L1, y)
b = vpshufb(L2, y ⊻ (x & 0b10000000))
c = a | b
Since the top bit is set in the input to either a or b, one of them is zerod, so
c represents the true result.

Now, because we constructed the mask such, we can do
m = 0x01 << ((x >>> 0x04) & 0b00000111)
if iszero(m & c), it passes
=#


##

# inverted 8 elems?

function findtype(x::ByteSet)
    if length(x) == 1
        return "all same"
    elseif Int(maximum(x) - minimum(x)) + 1 == length(x)
        return "within range"
    elseif maximum(x) - minimum(x) < 16
        return "within 16"
    elseif Int(maximum(~x) - minimum(~x)) + 1 == length(~x)
        return "inverted range"
    elseif maximum(~x) < 128
        return "inverted ascii"
    elseif maximum(x) < 128
        return "ascii"
    elseif maximum(~x) - minimum(~x) < 128
        return "inverted within 128"
    elseif maximum(x) - minimum(x) < 128
        return "within 128"
    elseif length(x) < 9
        return "elem8"
    else
        return "generic"
    end
end

## Testing
Base.rand(::Type{ByteSet}) = ByteSet(rand(UInt64), rand(UInt64), rand(UInt64), rand(UInt64))
Base.:~(x::ByteSet) = ByteSet(~x.a, ~x.b, ~x.c, ~x.d)

function get_simd_loops(machine::Machine)
    sets = ByteSet[]
    for node in traverse(machine.start)
        for (edge, dest) in node.edges
            # Only have self loops
            node === dest || continue

            # Only no-actions edges
            isempty(edge.actions) || continue
            push!(sets, edge.labels)
        end
    end
    return sets
end

vals = [0x08, 0x09, 0x14, 0x61, 0xa1, 0xb2]
bs = ByteSet(vals)
t1, t2 = bitmaps(bs, false)
input = v128(ntuple(x -> rand(vals), 16))

function qux(data::Vector{UInt8})
    p, len = 1, length(data)
    v = within128(load_v128(data, p), 0x7070707070f0f0f0f0f0f0f0f0f0f0e0, Val(0x00), Val(true))
    while (p < len) & haszerolayout(v)
        p += 16
        v = within128(load_v128(data, p), 0x7070707070f0f0f0f0f0f0f0f0f0f0e0, Val(0x00), Val(true))
    end
    # We use a while true loop even though there are at most 15 elements,
    # because we don't want the compiler to unroll this loop.
    i = 1
    @inbounds while true
        iszero(v.data[i].value) || break
        p += 1
        i += 1
    end
    return p
end

function qux(data::Vector{UInt8})
    p, p_end = 1, length(data)

    while true

        # Load and process a vector if there is at least one element left. This way, we guarantee
        # that the self-loop is passed after this while loop.
        vect = load_v128(data, p)
        v = within128(vect, 0x7070707070f0f0f0f0f0f0f0f0f0f0e0, Val(0x00), Val(true))

        # If the vector is nonzero or p is too high, we can't advance p by 16. We must check
        # each element of vector and check if we go out of bounds, then return.
        if !((p < p_end - 14) & haszerolayout(v))
            i = 1
            @inbounds while iszero(v.data[i].value) & (p ≤ p_end)
                i += 1 
                p += 1
            end
            break
        end
        p += 16
    end
    return p
end

cgc = Automa.CodeGenContext(checkbounds=false, generator=:goto, loopunroll=4)
cd = Automa.generate_exec_code(cgc, FASTQ.machine, FASTQ.actions)
open(x -> println(x, cd), "/tmp/machine.txt", "w")

import Automa
import Automa.RegExp: @re_str
const re = Automa.RegExp;

machine = let
    pat = re.rep1(re.space() \ re"\n")
    Automa.compile(pat)
end



function cmpp(data)
    seekstart(data)
    record = FASTA.Record()
    reader = FASTA.Reader(data)
    n = 0
    while !eof(data)
        read!(reader, record)
        n += length(record.sequence)
    end
    n
end

# Existing FASTA machine on 153 MB fasta:
# No actions: 78 ms (2 GB/s)
# Actions: 84 ms (1.8 GB/s)

# Existing_fasta
function appendfrom!(dst, dpos, src, spos, n)
    if length(dst) < dpos + n - 1
        resize!(dst, dpos + n - 1)
    end
    copyto!(dst, dpos, src, spos, n)
    return dst
end

actions = Dict(
    :mark => :(mark = p),
    :pos => :(pos = p),
    :countline => :(linenum += 1),
    :identifier => :(record.identifier = pos:p-1),
    :description => :(record.description = pos:p-1),
    :header => quote
        let n = p - mark
            appendfrom!(record.data, 1, data, mark, n)
            appendfrom!(record.data, filled + 1, b"\n", 1, 1)
            filled += n + 1
        end
    end,
    :letters => quote
        let n = p - mark
            appendfrom!(record.data, filled + 1, data, mark, n)
            if isempty(record.sequence)
                record.sequence = filled+1:filled+n
            else
                record.sequence = first(record.sequence):last(record.sequence)+n
            end
            filled += n
        end
    end,
    :record => quote
        record.filled = 1:filled
        if length(record.sequence) > 1000000
            println("foo")
            error()
        end
        tlen += length(record.sequence)
        filled = 0
        record.sequence = 1:0
        record.identifier = 1:0
        record.description = 1:0
    end
)

@eval function fasta_existing(data::Union{String,Vector{UInt8}})
    $(Automa.generate_init_code(cgc, FASTA.machine))
    record = FASTA.Record()
    pos = mark = filled = 0
    linenum = 1
    tlen = 0
    
    # p_end and p_eof were set to 0 and -1 in the init code,
    # we need to set them to the end of input, i.e. the length of `data`.
    p_end = p_eof = lastindex(data)
    
    # We just use an empty dict here because we don't want our parser
    # to do anything just yet - only parse the input
    $(code2)
    #$(Meta.parse(open(x -> String(read(x)), "/tmp/code2.jl")))

    # We need to make sure that we reached the accept state, else the 
    # input did not parse correctly
    iszero(cs) || error("failed to parse on byte ", p)

    return tlen
end;

## 91 ms

# Self loop 9 is main loop

code2 = Automa.generate_exec_code(cgc, FASTA.machine, actions)
open("/tmp/code2.jl", "w") do file
    println(file, code2)
end


@eval function fasta_new(data::Union{String,Vector{UInt8}})
    $(Automa.generate_init_code(cgc, FASTA.machine))
    record = FASTA.Record()
    pos = mark = filled = 0
    linenum = 1
    tlen = 0
    
    # p_end and p_eof were set to 0 and -1 in the init code,
    # we need to set them to the end of input, i.e. the length of `data`.
    p_end = p_eof = lastindex(data)
    
    # We just use an empty dict here because we don't want our parser
    # to do anything just yet - only parse the input


    $(Meta.parse(code))

    # We need to make sure that we reached the accept state, else the 
    # input did not parse correctly
    iszero(cs) || error("failed to parse on byte ", p)

    return tlen
end;

##########
struct Foo end
struct Bar end
Base.show(io::IO, ::Foo) = print(io, "Bar()") #NB!
x = Foo()
my_code = quote $x end
re_parsed = Meta.parse(string(my_code))
eval(my_code) == eval(re_parsed)


function vpshufb(data::Vector{UInt8}, selector::Vector{UInt8})
    result = Vector{UInt8}(undef, 16)
    for i in eachindex(result)
        selector_byte = selector[i]
        if iszero(selector_byte & 0x80)
            byte = data[(selector_byte & 0x0f) + 1]
        else
            byte = 0x00
        end
        result[i] = byte
    end
    return result
end

##################
import Automa: Action
# Try to generate some code
machine = FASTA.machine

if machine.start_state != 1
    throw(ArgumentError("Machine start state must be 1"))
end

# After actions we can go to a new action, a new state, or exit.
# This can be distinguished by the value of cs. 0 = exit.

# Get list of possible states following actions. action => (dest_state, jump_symbol)
following_states = Dict{Action, Set{Tuple{Int, Symbol}}}()
for node in traverse(machine.start)
    for (edge, destination) in node.edges
        for (i, action) in enumerate(edge.actions)
            jump_symbol = if i == length(edge.actions)
                Symbol("state_", destination.state)
            else
                Symbol("action_", edge.actions.actions[i+1].name)
            end
            push!(get!(following_states, action, Set{Tuple{Int, Symbol}}()), (destination.state, jump_symbol))
        end
    end
end
# add eof actions
for actions in values(machine.eof_actions)
    for (i, action) in enumerate(actions)
        jump_symbol = if i == length(actions)
            :exit
        else
            Symbol("action_", actions.actions[i+1].name)
        end
        push!(get(following_states, action, Set{Tuple{Int, Symbol}}()), (0, jump_symbol))
    end
end

# Here, a list of 



code = quote
    @goto state_1
end