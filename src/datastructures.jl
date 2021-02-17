import Base.length

Maybe{T} = Union{T, Nothing}

struct ListNode
    data
    next
end

mutable struct List
    root::Maybe{ListNode}
    leaf::Maybe{ListNode}
    len::UInt
end

function append!(l::List, value)::Nothing
    if l.root == nothing || l.leaf == nothing
        l.root = ListNode(value, nothing)
        l.leaf = root
    else
        l.leaf.data = value
        l.leaf.next = ListNode(nothing, nothing)
        l.leaf = l.leaf.next
    end
    l.len += 1
end

function length(l::List)::UInt
    l.len
end

function last(l)
    l.leaf.data
end

function empty_list()
    List(nothing, nothing, 0)
end
