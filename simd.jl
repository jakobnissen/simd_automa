module t

# This should be an Automa.Simd module that should be loaded
# All the vec_* (soon to be zero_) function should be exported
# export anything that is used in generated code!

# DONE # 0. Fix the code itself so it generates correct code with variable symbols
# DONE # 1. Rename everything to proper names
# DONE # 2. Figure out what to export and write export statements
# 3. Add imports SIMD/Libdl and add to Project.toml
# 4. Create the new module and copy-paste stuff into it.
# 5: Copy-paste tests in and add more. Also add tests for non-AVX2.
# 6. Run the lower-level tests to make sure the generated code works.
# 7. Now add a SIMD loop unroller


using SIMD
using Libdl
import Automa: ByteSet

const v256 = Vec{32, UInt8}
const v128 = Vec{16, UInt8}
const BVec = Union{v128, v256}
const _ZERO_v256 = v256(ntuple(i -> VecElement{UInt8}(0x00), 32))

# Discover if the system CPU has SSSE or AVX2 instruction sets
let
    llvmpaths = filter(lib -> occursin(r"LLVM\b", basename(lib)), Libdl.dllist())
    if length(llvmpaths) != 1
        throw(ArgumentError("Found multiple LLVM libraries"))
    end
    libllvm = Libdl.dlopen(llvmpaths[1])
    gethostcpufeatures = Libdl.dlsym(libllvm, :LLVMGetHostCPUFeatures)
    features_cstring = ccall(gethostcpufeatures, Cstring, ())
    features = split(unsafe_string(features_cstring), ',')
    Libc.free(features_cstring)
    @eval const SSSE3 = $(any(isequal("+ssse3"), features))
    @eval const AVX2 = $(any(isequal("+avx2"), features))
    @eval const SSE2 = $(any(isequal("+sse2"), features))
    @eval const DEFVEC = AVX2 ? v256 : v128
end

"""
    vpcmpeqb(a::BVec, b::BVec) -> BVec

Compare vectors `a` and `b` element wise and return a vector with `0x00`
where elements are not equal, and `0xff` where they are. Maps to the `vpcmpeqb`
AVX2 CPU instruction, or the `pcmpeqb` SSE2 instruction.
"""
function vpcmpeqb end

"""
    vpshufb(a::BVec, b::BVec) -> BVec

Maps to the AVX2 `vpshufb` instruction or the SSSE3 `pshufb` instruction depending
on the width of the BVec.
"""
function vpshufb end

"""
    vec_uge(a::BVec, b::BVec) -> BVec

Compare vectors `a` and `b` element wise and return a vector with `0xff`
where `a[i] ≥ b[i]``, and `0x00` otherwise. Implemented efficiently for CPUs
with the `vpcmpeqb` and `vpmaxub` instructions.

See also: [`vpcmpeqb`](@ref)
"""
function vec_uge end

let
    # icmp eq instruction yields bool (i1) values. We extend with sext to 0x00/0xff.
    # since that's the native output of vcmpeqb instruction, LLVM will optimize it
    # to just that.
    vpcmpeqb_template = """%res = icmp eq <N x i8> %0, %1
    %resb = sext <N x i1> %res to <N x i8>
    ret <N x i8> %resb
    """

    uge_template = """%res = icmp uge <N x i8> %0, %1
    %resb = sext <N x i1> %res to <N x i8>
    ret <N x i8> %resb
    """
    for N in (16, 32)
        T = NTuple{N, VecElement{UInt8}}
        ST = Vec{N, UInt8}
        instruction_set = N == 16 ? "ssse3" : "avx2"
        intrinsic = "llvm.x86.$(instruction_set).pshuf.b"
        vpcmpeqb_code = replace(vpcmpeqb_template, "<N x" => "<$(sizeof(T)) x")

        @eval @inline function vpcmpeqb(a::$ST, b::$ST)
            $(ST)(Base.llvmcall($vpcmpeqb_code, $T, Tuple{$T, $T}, a.data, b.data))
        end

        @eval @inline function vpshufb(a::$ST, b::$ST)
            $(ST)(ccall($intrinsic, llvmcall, $T, ($T, $T), a.data, b.data))
        end

        @eval const $(Symbol("_SHIFT", string(8N))) = $(ST)(ntuple(i -> 0x01 << ((i-1)%8), $N))
        @eval @inline bitshift_ones(shift::$ST) = vpshufb($(Symbol("_SHIFT", string(8N))), shift)

        uge_code = replace(uge_template, "<N x" => "<$(sizeof(T)) x")
        @eval @inline function vec_uge(a::$ST, b::$ST)
            $(ST)(Base.llvmcall($uge_code, $T, Tuple{$T, $T}, a.data, b.data))
        end
    end
end

@inline function leading_zero_bytes(v::v256)
    eqzero = vpcmpeqb(v, _ZERO_v256).data
    packed = ccall("llvm.x86.avx2.pmovmskb", llvmcall, UInt32, (NTuple{32, VecElement{UInt8}},), eqzero)
    @static if ENDIAN_BOM == 0x04030201
        return trailing_ones(packed)
    else
        return leading_ones(packed)
    end
end

# vpmovmskb requires AVX2, so we fall back to this.
@inline function leading_zero_bytes(v::v128)
    n = 0
    @inbounds for i in v.data
        iszero(i.value) || break
        n += 1
    end
    return n
end

@inline function haszerolayout(x::v128)
    return iszero(unsafe_load(Ptr{UInt128}(pointer_from_objref(Ref(x)))))
end

@inline function haszerolayout(x::v256)
    lower = iszero(unsafe_load(Ptr{UInt128}(pointer_from_objref(Ref(x))), 1))
    upper = iszero(unsafe_load(Ptr{UInt128}(pointer_from_objref(Ref(x))), 2))
    return lower & upper
end

@inline function loadvector(::Type{T}, p::Ptr) where {T <: BVec}
    unsafe_load(Ptr{T}(p))
end

# We have this to keep the same constant mask in memory.
@inline shrl4(x) = x >>> 0x04

Base.:~(x::ByteSet) = ByteSet(~x.a, ~x.b, ~x.c, ~x.d)
iscontiguous(x::ByteSet) = maximum(x) - minimum(x) == length(x) - 1

@inline function zerovec_generic(x::T, topzero::T, topone::T) where {T <: BVec}
    lower = vpshufb(topzero, x)
    upper = vpshufb(topone, x ⊻ 0b10000000)
    bitmap = lower | upper
    return bitmap & bitshift_ones(shrl4(x))
end

# If all values are within 128 of each other. We set offset to downshift values
# to 0x00:0x7f. If invert is set, this yields a vector of zeros if none of the inputs
# are within the shifted 0x00:0x7f range.
# If not inverted, all inputs with top bit will be set to 0x00, and then inv'd to 0xff.
# This will cause all shifts to fail.
# If inverted and ascii, we set offset to 0x80
@inline function zerovec_128(x::T, lut::T, offset::UInt8, f::Function) where {T <: BVec}
    y = x - offset
    bitmap = f(vpshufb(lut, y))
    return bitmap & bitshift_ones(shrl4(y))
end

@inline function zerovec_8elem(x::T, lut1::T, lut2::T) where {T <: BVec}
    # Get a 8-bit bitarray of the possible ones
    mask = vpshufb(lut1, x & 0b00001111)
    shifted = vpshufb(lut2, shrl4(x))
    return vpcmpeqb(shifted, mask & shifted)
end

# Here's one where they're 16 apart at most.
@inline function zerovec_16(x::T, lut::T, offset::UInt8) where {T <: BVec}
    y = x - offset
    lower = vpshufb(lut, y & 0b00001111)
    return lower | (y & 0b11110000) 
end

# One where it's a single range. After subtracting low, all values below end
# up above due to overflow and we can simply do a single ge check
@inline function zerovec_range(x::BVec, low::UInt8, len::UInt8)
    vec_uge((x - low), typeof(x)(len))
end

# One where, in all the disallowed values, the lower nibble is unique.
# This one is surprisingly common and very efficient.
# If all 0x80:0xff are allowed, the mask can be 0xff, and is compiled away
@inline function zerovec_inv_nibble(x::T, lut::T, mask::UInt8) where {T <: BVec}
    # If upper bit is set, vpshufb yields 0x00. 0x00 is not equal to any bytes with the
    # upper biset set, so the comparison will return 0x00, allowing it.
    return vpcmpeqb(x, vpshufb(lut, x & mask))
end


# Same as above, but inverted. Even better!
@inline function zerovec_nibble(x::T, lut::T, mask::UInt8) where {T <: BVec}
    return x ⊻ vpshufb(lut, x & mask)
end

# Simplest of all!
@inline zerovec_not(x::BVec, y::UInt8) = vpcmpeqb(x, typeof(x)(y))
@inline zerovec_same(x::BVec, y::UInt8) = x ⊻ y

function load_lut(::Type{T}, v::Vector{UInt8}) where {T <: BVec}
    T === v256 && (v = repeat(v, 2))
    return unsafe_load(Ptr{T}(pointer(v)))
end  

function generic_luts(::Type{T}, byteset::ByteSet, offset::UInt8, invert::Bool) where {
    T <: BVec}
    # If ascii, we set each allowed bit, but invert after vpshufb. Hence, if top bit
    # is set, it returns 0x00 and is inverted to 0xff, guaranteeing failure
    topzero = fill(invert ? 0xff : 0x00, 16)
    topone = copy(topzero)
    for byte in byteset
        byte -= offset
        # Lower 4 bits is used in vpshufb, so it's the index into the LUT
        index = (byte & 0x0f) + 0x01
        # Upper bit sets which of the two bitmaps we use.
        bitmap = (byte & 0x80) == 0x80 ? topone : topzero
        # Bits 5,6,7 from lowest control the shift. If, after a shift, the bit
        # aligns with a zero, it's in the bitmask
        shift = (byte >> 0x04) & 0x07
        bitmap[index] ⊻= 0x01 << shift
    end
    return load_lut(T, topzero), load_lut(T, topone)
end

function elem8_luts(::Type{T}, byteset::ByteSet) where {T <: BVec}
    allowed_mask = fill(0xff, 16)
    bitindices = fill(0x00, 16)
    for (i, byte) in enumerate(byteset)
        bitindex = 0x01 << (i - 1)
        allowed_mask[(byte & 0x0f) + 0x01] ⊻= bitindex
        bitindices[(byte >>> 0x04) + 0x01] ⊻= bitindex
    end
    return load_lut(T, allowed_mask), load_lut(T, bitindices)
end

function within16_lut(::Type{T}, byteset::ByteSet) where {T <: BVec}
    offset = minimum(byteset)
    lut = fill(0x01, 16)
    for byte in byteset
        lut[(byte - offset) + 1] = 0x00
    end
    return load_lut(T, lut)
end

function unique_lut(::Type{T}, byteset::ByteSet, invert::Bool) where {T <: BVec}
    # The default, unset value of the vector v must be one where v[x & 0x0f + 1] ⊻ x
    # is never accidentally zero.
    allowed = collect(0x01:0x10)
    for byte in (invert ? ~byteset : byteset)
        allowed[(byte & 0b00001111) + 1] = byte
    end
    return load_lut(T, allowed)
end 

########## Testing code below
function gen_zero_generic(sym::Symbol, x::ByteSet)
    lut1, lut2 = generic_luts(DEFVEC, x, 0x00, true)
    return :(zerovec_generic($sym, $lut1, $lut2))
end

function gen_zero_8elem(sym::Symbol, x::ByteSet)
    lut1, lut2 = elem8_luts(DEFVEC, x)
    return :(zerovec_8elem($sym, $lut1, $lut2))
end

function gen_zero_128(sym::Symbol, x::ByteSet, ascii::Bool, inverted::Bool)
    if ascii && !inverted
        offset, f, invert = 0x00, ~, false
    elseif ascii && inverted
        offset, f, invert = 0x80, ~, false
    elseif !ascii && !inverted
        offset, f, invert = minimum(x), ~, false
    else
        offset, f, invert = minimum(~x), identity, true
    end
    lut = generic_luts(DEFVEC, x, offset, invert)[1]
    return :(zerovec_128($sym, $lut, $offset, $f))
end

function gen_zero_16(sym::Symbol, x::ByteSet)
    lut = within16_lut(DEFVEC, x)
    return :(zerovec_16($sym, $lut, $(minimum(x))))
end

function gen_zero_range(sym::Symbol, x::ByteSet)
    return :(zerovec_range($sym, $(minimum(x)), $(UInt8(length(x)))))
end

function gen_zero_inv_range(sym::Symbol, x::ByteSet)
    # An inverted range is the same as a shifted range, because UInt8 arithmetic
    # is circular. So we can simply adjust the shift, and return regular vec_range
    return :(zerovec_range($sym, $(maximum(~x) + 0x01), $(UInt8(length(x)))))
end

function gen_zero_nibble(sym::Symbol, x::ByteSet, invert::Bool)
    lut = unique_lut(DEFVEC, x, invert)
    mask = maximum(invert ? ~x : x) > 0x7f ? 0x0f : 0xff
    if invert
        return :(zerovec_inv_nibble($sym, $lut, $mask))
    else
        return :(zerovec_nibble($sym, $lut, $mask))
    end
end

gen_zero_same(sym::Symbol, x::ByteSet) = :(zerovec_same($sym, $(minimum(x))))
gen_zero_not(sym::Symbol, x::ByteSet) = :(zerovec_not($sym, $(minimum(~x))))

# TODO: Make something useful of this.
function gencode(ressym::Symbol, sym::Symbol, x::ByteSet)
    if length(x) == 1
        expr = gen_zero_same(sym, x)
    elseif length(x) == 255
        return gen_zero_not(sym, x)
    elseif length(x) == length(Set([i & 0x0f for i in x]))
        expr = gen_zero_nibble(sym, x, false)
    elseif length(~x) == length(Set([i & 0x0f for i in ~x]))
        expr = gen_zero_nibble(sym, x, true)
    elseif iscontiguous(x)
        expr = gen_zero_range(sym, x)
    elseif iscontiguous(~x)
        expr = gen_zero_inv_range(sym, x)
    elseif maximum(x) - minimum(x) < 16
        expr = gen_zero_16(sym, x)
    elseif minimum(x) > 127
        expr = gen_zero_128(sym, x, true, true)
    elseif maximum(x) < 128
        expr = gen_zero_128(sym, x, true, false)
    elseif maximum(~x) - minimum(~x) < 128
        expr = gen_zero_128(sym, x, false, true)
    elseif maximum(x) - minimum(x) < 128
        expr = gen_zero_128(sym, x, false, false)
    elseif length(x) < 9
        expr = gen_zero_8elem(sym, x)
    else
        expr = gen_zero_generic(sym, x)
    end
    return :($ressym = $(expr))
end

###
function test_function(f::Function, bs::ByteSet)
    pass = true
    for i in 0x00:0x0f
        v = f(load_lut(DEFVEC, collect((0x00:0x0f))) + (UInt8(i) * 0x10))
        for j in 0x00:0x0f
            n = (0x10 * i) + j
            pass &= ((n in bs) == (v[j+1] == 0x00))
        end
    end
    return pass
end

bs_same = ByteSet([0x07])
bs_not = ~bs_same
bs_nibble = ByteSet([0x02, 0x0a, 0x1b, 0x1c, 0x1d, 0x20, 0x7e])
bs_inv_nibble = ~bs_nibble
bs_range = ByteSet(0xa9:0xc1)
bs_inv_range = ByteSet([0x00:0x09; 0x4a:0xff])
bs_16 = ByteSet([0x45, 0x48, 0x49, 0x50, 0x55, 0x53])
bs_inv_ascii = ByteSet(rand(0x8a:0xf1, 50))
bs_ascii = ByteSet(rand(0x0a:0x61, 50))
bs_inv_128 = ~ByteSet(rand(0x31:0xa1, 50))
bs_128 = ByteSet(rand(0x31:0xa1, 50))
bs_8elem = ByteSet(rand(0x00:0xff, 8))
bs_generic = ByteSet(rand(0x00:0xff, 75))

for symbol in [:same, :not, :nibble, :inv_nibble, :range, :inv_range,
    Symbol(:16), :inv_ascii, :ascii, :inv_128, Symbol(:128), Symbol("8elem"), :generic]
    #@eval function $(Symbol("f_", string(symbol)))(x::DEFVEC)
    bs = @eval $(Symbol("bs_", string(symbol)))
    @eval function $(Symbol("f_", string(symbol)))(x::DEFVEC)
        $(gencode(:y, :x, bs))
    end
    r = @eval test_function($(Symbol("f_", string(symbol))), $(Symbol("bs_", string(symbol))))
    if !r
        error("Function $symbol failed!")
    end
end

# Note: This function has been carefully crafted to produce (nearly) optimal
# assembly code for AVX2-capable CPUs. Change with great care.
function foo_code(xsym::Symbol, ysym::Symbol, byteset::ByteSet)
    return quote
        while true
            $xsym = loadvector($DEFVEC, pointer(data, p))
            $(gencode(ysym, xsym, byteset))
            if !haszerolayout(y) || p > p_end - $(sizeof(DEFVEC))
                p = min(p_end + 1, p + leading_zero_bytes(y))
                break
            end
            p += $(sizeof(DEFVEC))
        end
    end
end

@eval function foo(data::Vector{UInt8}, p::Int)
    p_end = length(data)
    $(foo_code(:x, :y, bs_generic))
    return p
end

# We need to export all names that can end up in the generated code.
# else the generated code will lead to NameError
export leading_zero_bytes,
    haszerolayout,
    loadvector,
    zerovec_128,
    zerovec_16,
    zerovec_8elem,
    zerovec_generic,
    zerovec_inv_nibble,
    zerovec_nibble,
    zerovec_not,
    zerovec_range,
    zerovec_same

end # t


