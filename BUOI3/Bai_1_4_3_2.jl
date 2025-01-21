# queue_julia.jl

mutable struct Queue
    elements::Vector{Any} # Vector chứa các phần tử

    # Constructor khởi tạo hàng đợi rỗng sử dụng `new`
    Queue() = new(Vector{Any}())
end

# Hàm thêm phần tử vào hàng đợi
function enqueue!(q::Queue, item)
    push!(q.elements, item)
    println("Đã thêm '$item' vào hàng đợi.")
end

# Hàm loại bỏ phần tử khỏi hàng đợi
function dequeue!(q::Queue)
    if !isempty(q.elements)
        item = popfirst!(q.elements)
        println("Đã lấy '$item' ra khỏi hàng đợi.")
        return item
    else
        println("Hàng đợi trống!")
        return nothing
    end
end
# Hàm xem phần tử ở đầu hàng đợi
function front(q::Queue)
    if !isempty(q.elements)
        return q.elements[1]
    else
        println("Hàng đợi trống!")
        return nothing
    end
end

# Hàm kiểm tra hàng đợi rỗng
function is_empty(q::Queue)
    return isempty(q.elements)
end

# Hàm in nội dung hàng đợi
function display(q::Queue)
    println("Hàng đợi (đầu đến cuối): ", q.elements)
end
# Minh họa sử dụng hàng đợi
function main()
    queue = Queue()
    enqueue!(queue, "Tài Liệu 1")
    enqueue!(queue, "Tài liệu 2")
    enqueue!(queue, "Tài liệu 3")
    display(queue) # Output: Hàng đợi (đầu đến cuối): ["Tài Liệu 1", "Tài Liệu 2", "Tài Liệu 3"]

    front_item = front(queue)
    println("Phần tử ở đầu hàng đợi: ", front_item) # Output: Tài Liệu 1

    dequeue!(queue)
    display(queue) # Output: Hàng đợi (đầu đến cuối): ["Tài Liệu 2", "Tài Liệu 3"] 

    println("Hàng đợi có trống không? ", is_empty(queue) ? "Có" : "Không") # Output: Không
end

main()