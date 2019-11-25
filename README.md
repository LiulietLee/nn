# 栈的基本操作

## 实验目的
熟练掌握栈这种抽象数据类型，能在相应的应用问题中正确选用。熟练掌握栈类型的两种实现方法，两种存储结构表和基本操作实现算法，注意栈空和栈满的条件及它们的描述方法。

## 实验内容
栈的数据结构定义如下：
```cpp
#define stack struct Stack
#define STACK_POP_ERR 42

struct Stack {
    int val[100]; // 栈空间
    int top;      // 栈顶
};
```
请依据栈的定义完成判断栈是否为空的函数 `empty`：
```cpp
bool empty(stack *stk) {
  // 请完成这个函数
  
}
```
堆栈使用两种基本操作：推入（压栈，push）和弹出（弹栈，pop）：
![asdf](https://github.com/LiulietLee/nn/blob/master/stack.png)
- 推入：将数据放入堆栈顶端，堆栈顶端移到新放入的数据。依据上面的栈的定义请完成下面的 `push` 函数：
```cpp
void push(stack *stk, int x) {
  // 请完成这个函数
  
}
```
- 弹出：将堆栈顶端数据移除，堆栈顶端移到移除后的下一笔数据。依据上面栈的定义请完成下面的 `pop` 函数：
```cpp
int pop(stack *stk) {
  // 请完成这个函数
  
}
```

## 问题实践

输入一个括号序列，要求用上面完成的栈的各种方法判断是否括号是匹配的。
例如 `()()(())` 是匹配的，`(()()(` 是不匹配的。
如果括号序列是匹配的请返回 true，否则返回 false。

```cpp
bool isMatch(const char *bracketSequence, int sequenceLength) {
  stack stk;
  stk.top = 0;
  // 请完成这个函数

}
```

## 实验总结
