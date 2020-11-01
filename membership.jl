module t

import Automa.ByteSet

Base.:~(x::ByteSet) = ByteSet(~x.a, ~x.b, ~x.c, ~x.d)
iscontiguous(x::ByteSet) = maximum(x) - minimum(x) == length(x) - 1

# Recursively build code
function make_membership_code(sym::Symbol, x::ByteSet)
    if length(x) == 1
        return (1, :($sym == $(minimum(x))))
    elseif iscontiguous(x)
        return (2, :(in($sym, $(minimum(x)):$(maximum(x)))))
    elseif maximum(x)-minimum(x) < 64
        min = minimum(x)
        bitmap = UInt64(0)
        for i in x
            bitmap |= UInt64(1) << (i - min)
        end
        return (3, :(UInt64(1) & ($bitmap >>> $sym) == UInt64(1)))
    else
        return (4, nothing)
    end 
end

"Peel off the first 64 values from the minimum to a new byteset"
function peel(x::ByteSet)
    min = minimum(x)
    if min > 191
        return (x, ByteSet())
    end
    mask = UInt64(1) << UInt64(min & 63) - UInt64(1)
    if min < 64
        peeled = ByteSet(x.a, x.b & mask, 0, 0)
        rest = ByteSet(0, x.b & ~mask, x.c, x.d)
    elseif min < 128
        peeled = ByteSet(0, x.b, x.c & mask, 0)
        rest = ByteSet(0, 0, x.c & ~mask, x.d)
    else
        peeled = ByteSet(0, 0, x.c, x.d & mask)
        rest = ByteSet(0, 0, 0, x.d & ~mask)
    end
    return peeled, rest
end

function foo(sym::Symbol, x::ByteSet)
    p1, c1 = make_membership_code(sym, x)
    p2, c2 = make_membership_code(sym, ~x)
    expr = p2 < p1 ? :(!($c2)) : c1 
    pri = min(p1, p2)
    if pri < 4
        return expr
    else
        head, tail = peel(x)
        return :($(make_membership_code(sym, head)[2]) || $(foo(sym, tail)))
    end
end

end # module
