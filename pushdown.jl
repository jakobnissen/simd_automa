# Make this macro exist!
#=
@pda begin
    tree = subtree * ";"
    subtee = name | internal
    name = "[A-Z]*"
    internal = "(" * branchset * ")" * name
    branchset = subtree | (subtree * "," * branchset)
end
=#
#= NFA optimizations
If you have a self-epsilon pop :s edge, replace it with popping as many :s as possible
You can simply remove self-epsilon push edges, as long as you have the above optimization
=#

#=
Algorithm to create NFA
First, create nodes for each symbol. Each symbol has a start node (labeled s below), an
end node (labeled e). At each concatenation, add another node 1..N.

Now, look at "tree". At tree s node, there is subtree, so draw an eps edge from tree_s to
subtree_s. Since we go into the subtree_s node we enter subtree, so push subtree onto stack.

Since after subtree is tree_1, draw an edge from subtree_e to tree_1, also eps edge.
Finally, draw an edge from tree_1 to tree_e, with the byteset [UInt8(';')].

Next, look at subtree. After subtree_s, we can have name_s and internal_s, so draw eps edges
to these. Now, where can we go after subtree_e? Well, "subtree" appears in the symbols tree and
branchset. So draw eps edges to tree_1, branchset_1 and branchset_e. Whenever we move to a node
with the _s suffix, we end a pattern, so e.g. on the edge subtree_e -> branchset_e, we pop branchset
symbol off the stack.


           s       1    e
    tree = subtree * ";"

             s  e   s      e
    subtee = name | internal

    s e s      1     e
    name = "" | ("A-Z" * name)

               s   1           2     3     e
    internal = "(" * branchset * ")" * name
     
                s      e  s        1     2          e
    branchset = subtree | (subtree * "," * branchset)

# Hypothetical pattern
       s   s  1     3   4      end
foo = (x | (y * m)) * z * (w | v)
=#

#NFA diagram
unoptimized = """
digraph {
graph [ rankdir = LR ];

start [ shape = point ];
tree_s [ shape = circle ];
tree_e [ shape = doublecircle ];
tree_1 [ shape = circle ];
subtree_s [ shape = circle ];
subtree_e [ shape = circle ];
name_s [ shape = circle ];
name_1 [ shape = circle ];
name_e [ shape = circle ];
internal_s [ shape = circle ];
internal_1 [ shape = circle ];
internal_2 [ shape = circle ];
internal_3 [ shape = circle ];
internal_e [ shape = circle ];
branchset_s [ shape = circle ];
branchset_e [ shape = circle ];
branchset_1 [ shape = circle ];
branchset_2 [ shape = circle ];

start -> tree_s [ label = "ϵ/⬇tree" ];
tree_s -> subtree_s [ label = "ϵ/⬇subtree" ];
tree_1 -> tree_e [ label = "\";\"/⬆tree" ];

subtree_s -> name_s [ label = "ϵ/⬇name" ];
subtree_s -> internal_s [ label = "ϵ/⬇internal" ];
subtree_e -> branchset_e [ label = "ϵ/⬆branchset" ];
subtree_e -> branchset_1 [ label = "ϵ"];
subtree_e -> tree_1 [ label = "ϵ"];

name_s -> name_e [ label = "ϵ/⬆name" ];
name_s -> name_1 [ label = "\"A-Z\"" ];
name_1 -> name_s [ label = "ϵ/⬇name" ];
name_e -> name_e [ label = "ϵ/⬆name" ];
name_e -> subtree_e [ label = "ϵ/⬆subtree" ];
name_e -> internal_e [ label = "ϵ/⬆internal" ];

internal_s -> internal_1 [ label = "\"(\"" ];
internal_1 -> branchset_s [ label = "ϵ/⬇branchset" ];
internal_2 -> internal_3 [ label = "\")\"" ];
internal_3 -> name_s [ label = "ϵ/⬇name" ];
internal_e -> subtree_e [ label = "ϵ/⬆subtree" ];

branchset_s -> subtree_s [ label = "ϵ/⬇subtree" ];
branchset_1 -> branchset_2 [ label = "\",\"" ];
branchset_2 -> branchset_s [ label = "ϵ/⬇branchset" ];
branchset_e -> internal_2 [ label = "ϵ"];
branchset_e -> branchset_e [ label = "ϵ/⬆branchset" ];
}
"""

# I optimized this by hand
# Here, ⬆? means pop if applicable, and ⬆+ means pop all symbols.
#= Algo:
function TRAVERSE(NODE)
    Move along all possible epsilon-paths from NODE until you find an edge with some bytes on it.
    Keep track of what you push and pop on the stack meanwhile.
    You are not allowed to move through an edge that pops a symbol that is not at the top of the stack.
    If NODE is starting node, you know the top of the stack is empty before you started at NODE.
    Therefore, you can't move through popping edges if the stack is empty.
    If NODE is not starting node, you don't know what's on the stack and can move through popping nodes.
    For each path, return the final byteset of the edge you stopped on, as well as the final "state" of the
    stack. This state can be "negative", e.g. a "total" of 5 pops.

VISITED = Set()
REMAINING = Set([start])
while !isempty(REMAINING)
    T = TRAVERSE(pop!(REMAINING)) \ VISITED
    union!(REMAINING, T)
    union!(VISITED, T)
end

Now find identical nodes. It's most efficient to see if the only possible path from a start
is eps path to an existing path. Otherwise, nodes are identical if they have the same paths leading out
with same actions and ending bytesets. Remove the identical nodes.

If you move to an e-edge with a self-eps edge that pops e.g, that's gives a ⬆+-action, i.e. pop as many
symbols of that type as possible.
The only "magic" (non-algorithmic thinking I did) was that, in the NFA diagram, one can move from name_e to subtree_e through two paths. This should yield ⬆?internal,⬆subtree.
Not sure how to make an algorithm that realizes that.
=#
optimized = """
digraph {
graph [ rankdir = LR ];
start [ shape = point ];
STATE1 [ shape = doublecircle ];
INT1 [ shape = circle ];
INT3 [ shape = doublecircle ];
EOF_START [ shape = point ];
EOF_INT3 [ shape = point ];

start -> STATE1;

STATE1 -> INT1 [ label = "\"(\"/⬇tree,⬇subtree,⬇internal" ];
STATE1 -> INT3 [ label = "\"[A-Z]\"/⬇tree,⬇subtree,⬇name" ];
STATE1 -> EOF_START [ label = "\";\"/EOF", style = dashed ];

INT1 -> INT1 [ label = "\",\"/⬇branchset" ];
INT1 -> INT1 [ label = "\"(\"/⬇branchset,⬇subtree,⬇internal" ];
INT1 -> INT3 [ label = "\"[A-Z]\"/⬇branchset,⬇subtree,⬇name" ];
INT1 -> INT3 [ label = "\")\"/⬆+branchset" ];

INT3 -> INT1 [ label = "\",\"/⬆+name,⬆?internal,⬆subtree" ]
INT3 -> EOF_INT3 [ label = "\";\"/⬆+name,⬆?internal,⬆subtree,⬆+tree,EOF", style = dashed ];
INT3 -> INT3 [ label = "\"[A-Z]\"/⬇name" ];
INT3 -> INT3 [ label = "\")\"/⬆+name,⬆?internal,⬆subtree,⬆+branchset" ];
}
"""

mutable struct Stack
    pos::Int
    vec::Vector{Int32}
    symbols::Vector{Symbol}
end

# State, byte, expected, found
struct StackError <: Exception
    msg::String
end
@noinline function throw_stackerror(expected::Symbol, found::Symbol)
     throw(StackError("Popped symbol $(found), expected $(expected)"))
end

@noinline throw_emptystack() = throw(StackError("Empty stack"))

function Base.push!(stack::Stack, sym::Int32)
    stack.pos == length(stack.vec) && resize!(stack.vec, 2*length(stack.vec) % UInt)
    stack.pos += 1
    @inbounds stack.vec[stack.pos] = sym
    return nothing
end

function Base.pop!(stack::Stack, expected::Int32)
    iszero(stack.pos) && return Int32(-1)
    found = @inbounds stack.vec[stack.pos]
    stack.pos -= 1
    return found
end
