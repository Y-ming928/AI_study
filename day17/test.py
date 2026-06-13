class Demo:
    def __init__(self):
        self.x = 10

    def foo(self):
        pass

d = Demo()

# 检查对象d是否有属性'x'
print(hasattr(d, 'x'))  # True

# 检查对象d是否有方法'foo'
print(hasattr(d, 'foo'))  # True

# 检查对象d是否有属性'y'
print(hasattr(d, 'y'))  # False
