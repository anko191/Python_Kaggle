# property decorator
# https://www.headboost.jp/python-property/
class Test():
    """init method"""
    def __init__(self):
        self.__name = "Phan" # 最初に__を付けると非公開変数参照できない

    """クラスブロック内で非公開変数"""
    def myname(self):
        return self.__name
    """get_name method"""
    def get_name(self):
        return self.__name
    """set_name method"""
    def set_name(self, value):
        self.__name = value
    """
    getter:インスタンスの値を返す
    setter:インスタンス変数に新しい値を設定する
    """
    """name property's getter"""
    @property
    def name(self):
        return self.__name
    """name property 's setter"""
    @name.setter
    def name(self, value):
        self.__name = value
test = Test()
print(test.name)
test.name = 'Phyooo'
# こうじゃないと変更が出来ないのかなぁああ？？
print(test.name)