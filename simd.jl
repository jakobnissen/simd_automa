module t

# This should be an Automa.Simd module that should be loaded

using SIMD
using Libdl
using Base: llvmcall
import Automa: ByteSet

const v256 = Vec{32, UInt8}
const v128 = Vec{16, UInt8}
const BVec = Union{v128, v256}
const t256 = NTuple{32, VecElement{UInt8}}
const t128 = NTuple{16, VecElement{UInt8}} 

# Discover if the system has SSSE or AVX2
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
    @eval const DEFVEC = AVX2 ? v256 : v128
end

@inline function vpshufb(x::v256, mask::v256)
    v256(ccall("llvm.x86.avx2.pshuf.b", llvmcall, t256, (t256, t256), x.data, mask.data))
end

@inline function vpshufb(x::v128, mask::v128)
    v128(ccall("llvm.x86.ssse3.pshuf.b.128", llvmcall, t128, (t128, t128), x.data, mask.data))
end

const _SHIFT128 = v128(ntuple(i -> 0x01 << ((i-1)%8), 16))
@inline bitshift_ones(shift::v128) = vpshufb(_SHIFT128, shift)

const _SHIFT256 = v256(ntuple(i -> 0x01 << ((i-1)%8), 32))
@inline bitshift_ones(shift::v256) = vpshufb(_SHIFT256, shift)

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

@inline function toUInt8(x::Vec{N, Bool}) where N
    unsafe_load(Ptr{Vec{N, UInt8}}(pointer_from_objref(Ref(x))))
end

# We have this to keep the same constant mask in memory.
@inline shrl4(x) = x >>> 0x04

Base.:~(x::ByteSet) = ByteSet(~x.a, ~x.b, ~x.c, ~x.d)
iscontiguous(x::ByteSet) = maximum(x) - minimum(x) == length(x) - 1

@inline function vec_generic(x::T, topzero::T, topone::T) where {T <: BVec}
    y = x & 0b10001111
    lower = vpshufb(topzero, y)
    upper = vpshufb(topone, y ⊻ 0b10000000)
    bitmap = lower | upper
    return bitmap & bitshift_ones(shrl4(x))
end

# If all values are within 128 of each other. We set offset to downshift values
# to 0x00:0x7f. If invert is set, this yields a vector of zeros if none of the inputs
# are within the shifted 0x00:0x7f range.
# If not inverted, all inputs with top bit will be set to 0x00, and then inv'd to 0xff.
# This will cause all shifts to fail.
# If inverted and ascii, we set offset to 0x80
@inline function vec_within128(x::T, lut::T, offset::UInt8, invert::Bool) where {T <: BVec}
    f = invert ? identity : Base.:~
    bitmap = f(vpshufb(lut, x - offset))
    return bitmap & bitshift_ones(shrl4(x))
end

@inline function vec_8elem(x::T, lut1::T, lut2::T) where {T <: BVec}
    # Get a 8-bit bitarray of the possible ones
    mask = vpshufb(lut1, x & 0b00001111)
    shifted = vpshufb(lut2, shrl4(x))
    return toUInt8((mask & shifted) == 0x00)
end

# Here's one where they're 16 apart at most.
@inline function vec_within16(x::T, lut::T, offset::UInt8) where {T <: BVec}
    y = x - offset
    lower = vpshufb(lut, y & 0b00001111)
    return lower | (y & 0b11110000) 
end

# One where it's a single range. After subtracting low, all values below end
# up above due to overflow and we can simply do a single le check
@inline function vec_range(x::BVec, low::UInt8, len::UInt8)
    toUInt8((x - low) < len)
end

# Simplest of all!
@inline vec_same(x::BVec, y::UInt8) = x ⊻ y

function load_lut(::Type{T}, v::Vector{UInt8}) where {T <: BVec}
    v = repeat(v, div(sizeof(T), 16))
    return unsafe_load(Ptr{T}(pointer(v)))
end  

function generic_luts(::Type{T}, byteset::ByteSet, offset::UInt8, ascii::Bool) where {
    T <: BVec}
    # If ascii, we set each allowed bit, but invert after vpshufb. Hence, if top bit
    # is set, it returns 0x00 and is inverted to 0xff, guaranteeing failure
    topzero = fill(ascii ? 0x00 : 0xff, 16)
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
    return load_lut(T, topzero), load_lut(T, topzero)
end

function elem8_luts(::Type{T}, byteset::ByteSet) where {T <: BVec}
    lower = fill(0x00, 16)
    upper = copy(lower)
    for (i, byte) in enumerate(byteset)
        bitindex = 0x01 << (i - 1)
        lower[(byte & 0x0f) + 0x01] ⊻= bitindex
        upper[(byte >>> 0x04) + 0x01] ⊻= bitindex
    end
    return load_lut(T, lower), load_lut(T, upper)
end

function within16_lut(::Type{T}, byteset::ByteSet) where {T <: BVec}
    offset = minimum(byteset)
    lut = fill(0x01, 16)
    for byte in byteset
        lut[(byte - offset) + 1] = 0x00
    end
    return load_lut(T, lut)
end
    

########## Testing code below
function make_generic_veccode(x::ByteSet)
    lut1, lut2 = generic_luts(DEFVEC, x, 0x00, false)
    return :(vec_generic(x, $lut1, $lut2))
end

function make_8elem_veccode(x::ByteSet)
    lut1, lut2 = elem8_luts(DEFVEC, x)
    return :(vec_8elem(x, $lut1, $lut2))
end

function make_128_veccode(x::ByteSet, ascii::Bool, inverted::Bool)
    offset = if ascii
        if inverted
            0x80
        else
            0x00
        end
    else
        minimum(inverted ? ~x : x)
    end
    lut = generic_luts(DEFVEC, x, offset, inverted)[1]
    return :(vec_within128(x, $lut, $offset, $(inverted & !ascii)))
end

function make_within16_code(x::ByteSet)
    lut = within16_lut(DEFVEC, x)
    return :(vec_within16(x, $lut, $(minimum(x))))
end

function make_range_code(x::ByteSet)
    return :(vec_range(x, $(minimum(x)), $(UInt8(length(x)))))
end

function make_inv_range_code(x::ByteSet)
    # An inverted range is the same as a shifted range, because UInt8 arithmetic
    # is circular. So we can simply adjust the shift, and return regular vec_range
    return :(vec_range(x, $(maximum(~x) + 0x01), $(UInt8(length(x)))))
end

make_allsame_code(x::ByteSet) = :(vec_same(x, $(minimum(x))))


# TODO: Make something useful of this.
function gencode(x::ByteSet)
    if length(x) == 1
        return return make_allsame_code(x)
    elseif iscontiguous(x)
        return make_range_code(x)
    elseif iscontiguous(~x)
    # Inverted range
        return make_inv_range_code(x)
    elseif maximum(x) - minimum(x) < 16
        return make_within16_code(x)
    # Inverted ASCII
    elseif minimum(x) > 127
        return make_128_veccode(x, true, true)
    # Ascii
    elseif maximum(x) < 128
        return make_128_veccode(x, true, false)
    # Inverted within 128
    elseif maximum(~x) - minimum(~x) < 128
        return make_128_veccode(x, false, true)
    # Within 128
    elseif maximum(x) - minimum(x) < 128
        return make_128_veccode(x, false, false)
    elseif length(x) < 9
        return make_8elem_veccode(x)
    else
        return make_generic_veccode(x)
    end
end


###
bs_same = ByteSet([0x07])
@eval function f_same(x)
    y = $(gencode(bs_same))
end

bs_range = ByteSet(0xa9:0xc1)
@eval function f_range(x)
    y = $(gencode(bs_range))
end

bs_inv_range = ByteSet([0x00:0x09; 0x4a:0xff])
@eval function f_inv_range(x)
    y = $(gencode(bs_inv_range))
end

bs_within_16 = ByteSet([0x45, 0x48, 0x49, 0x50, 0x52, 0x53])
@eval function f_witin16(x)
    y = $(gencode(bs_within_16))
end

bs_inv_ascii = ByteSet(rand(0x8a:0xf1, 50))
@eval function f_inv_ascii(x)
    y = $(gencode(bs_inv_ascii))
end

bs_ascii = ByteSet(rand(0x0a:0x61, 50))
@eval function f_ascii(x)
    y = $(gencode(bs_ascii))
end
end # t
