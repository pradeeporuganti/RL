def typed_property(name, expected_type):
    storage_name = '_' + name
    
    @property
    def prop(self):
        return getattr(self, storage_name)
    
    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a {}'.format(name, expected_type))
        setattr(self, storage_name, value)
        return prop

    #@prop.getter
    #def prop(self):
    #    return prop

class Test():
    a = typed_property('a', int)
    b = typed_property('b', int)

    def __init__(self, a = None, b = None, c = None) -> None:
        self._a = a
        self._b = b
   
def main():
    j = Test()
    j.a = 10
    j.b = j.a
    print(j.a); print(j.b); 

if __name__ == '__main__':
    main()
    