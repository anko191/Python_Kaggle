a: int
a = None
# エラー出ねえｗｗ
# 本来はエラー出るってよ～～
# print(type(a))

"""Union型"""
from typing import Union
b: Union[int, None]
# Union を使うことによって、2つの方が代入される可能性があることを明示できます
b = None

"""Optional"""
from typing import Optional
c: Optional[int]
c = None

"""cast"""
from typing import Optional
a: Optional[int] = 0
def add_one(x: int) -> int:
    return x + 1

print(add_one(a))

# train_weight: tp.Optional[np.ndarray] = None,
# val_weight: tp.Optional[np.ndarray] = None,