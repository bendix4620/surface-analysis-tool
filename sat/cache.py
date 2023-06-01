from __future__ import annotations
from typing import Any, Callable, Sequence, Union, Type
from collections import OrderedDict
from functools import wraps
from abc import ABC, abstractmethod

EMPTY = object() # sentinel value for empty cache entries


def cached(mth):
    """Cache decorator"""
    key = mth.__name__

    @wraps(mth)
    def wrapper(self, *args, **kwargs):
        entry = self.cache.getEntry(key)
        if entry.isempty():
            entry.val = mth(self, *args, **kwargs)  # bypass cache update
        return entry.get()
    return property(wrapper)


class CacheHirarchy(ABC):
    def __init__(self,
            name: str = None,
            src: Sequence[Type[CacheHirarchy]] = [], 
            dst: Sequence[Type[CacheHirarchy]] = []
        ):
        """A cache object with cache invalidation hirarchy
        
        Parameters
        ----------
        name: str
            Display Name
        src: Sequence of DependendCache Subclasses
            Sources that self depends on
        dst: Sequence of DependendCache Subclasses
            Destinations that depend on self
        """
        if name is None:
            name = type(self).__name__
        self.name = name
        
        self.master = None
        self.destinations = []
        self.add_src(*src)
        self.add_dst(*dst)
        return

    def add_src(self, *src: Type[CacheHirarchy]) -> Type[CacheHirarchy]:
        """Add a cache dependency"""
        for s in src:
            if not isinstance(s, CacheHirarchy):
                raise TypeError("Can only link to other objects of type "
                                f"CacheHirarchy, not {type(s)}")
            s.add_dst(self)
        return self
    
    def add_dst(self, *dst: Type[CacheHirarchy]) -> Type[CacheHirarchy]:
        """Add cache that depends on self"""
        for d in dst:
            if not isinstance(d, CacheHirarchy):
                raise TypeError("Can only link to other objects of type "
                                f"CacheHirarchy, not {type(d)}")
            self.destinations.append(d)
        return self

    def invalidate(self):
        """Invalidate self and destinations"""
        self.invalidate_dst()

    def invalidate_dst(self):
        """Invalidate destinations"""
        for dst in self.destinations:
            dst.invalidate()
        if self.master is not None:
            self.master.invalidate_dst()

    def cascade(self) -> tuple:
        """Return the dependency cascade starting with self"""
        # use tuple(str, []) instead of dict to allow non-distinct keys
        cascade = [] 
        for dst in self.destinations:
            cascade.append(dst.cascade())
        return (type(self).__name__ + " " + self.name, cascade)


class Entry(CacheHirarchy):
    """Cache entries that can be linked to invalidate dependent Entry
    instances on update automatically
    """

    @wraps(CacheHirarchy.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val = EMPTY
    
    @wraps(CacheHirarchy.invalidate)
    def invalidate(self):
        self.val = EMPTY
        return super().invalidate()
        
    def get(self) -> Any:
        """Return the cached value"""
        return self.val

    def set(self, val: Any):
        """Update the cached value"""
        self.val = val
        self.invalidate_dst()

    def delete(self):
        """Delete the cached value via the assiciated deleter function
        Do NOT call this function from inside the deleter
        """
        
        self.invalidate()
    
    def isempty(self) -> bool:
        """Return True if this Entry has a value, False if not"""
        return self.val is EMPTY
    
    def __repr__(self):
        if self.isempty():
            return f"Empty Entry '{self.name}'"
        return f"Entry '{self.name}' = {self.val}"


class Cache(CacheHirarchy, OrderedDict):
    """Cache dictionary to hold items of Entry type"""
    
    @wraps(CacheHirarchy.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__()

    def invalidate(self):
        """Invalidate all stored entries"""
        for key in self.keys():
            self.getEntry(key).invalidate()
        return super().invalidate()

    def __getitem__(self, __key: Any) -> Any:
        """Get value of the Entry"""
        return super().__getitem__(__key).get()
    
    def __setitem__(self, __key: str, __value: Any):
        """Set value of the Entry"""
        super().__getitem__(__key).set(__value)
        return

    def __delitem__(self, __key: Any):
        """Delete value of the Entry"""
        return super().__getitem__(__key).delete()

    def getEntry(self, key: str) -> Entry:
        """Get Entry instance"""
        return super().__getitem__(key)
    
    def setEntry(self, key: str, entry: Entry) -> Entry:
        """Set an Entry instance"""
        if not isinstance(entry, Entry):
            raise TypeError(f"Expected Entry type, got {type(entry)}")
        entry.master = self # assign self as master
        super().__setitem__(key, entry)
        return
    
    def deleteEntry(self, key: str):
        """Delete Entry instance"""
        return super().__delitem__(key)
    
    def newEntry(self,
            key: str = "", 
            name: str = None, 
            src: Sequence[Type[CacheHirarchy]] = [], 
            dst: Sequence[Type[CacheHirarchy]] = []
        ):
        """Set and return a new Entry instance"""
        entry = Entry(name=name, src=src, dst=dst)
        self.setEntry(key, entry)
        return entry
    
    def is_compatible(self, master, asbool=False):
        """Check the cache entry compatibility with master
        Return compatibility check as boolean value if asbool is True,
        otherwise raise an AttributeError
        """
        for key in self.keys():
            if key not in dir(master):
                if not asbool:
                    msg = (f"No method {type(master).__name__}{key} found for "
                           f"Entry {self.getEntry(key).name}")
                    raise AttributeError(msg)
                return False
        return True


if __name__ == "__main__":
    CacheHirarchy()
    Entry()
    Cache()
    class Test:
        def __init__(self):
            self.cache = Cache()
            foo = self.cache.newEntry(key="foo", name="FOO")

            bar = Entry(name="BAR", src=[foo])
            self.cache.setEntry("bar", bar)

        @cached
        def foo(self):
            print("Called foo")
            return "foo"
        
        @foo.setter
        def foo(self, val):
            self.cache["foo"] = val

        @cached
        def bar(self):
            print("Called bar")
            return str(self.foo) + "-bar"
        
        @cached
        def derived(self):
            return "not" + str(self.bar)
        
    
    test = Test()

    # Called foo
    print(test.foo) # foo
    print(test.foo) # foo
    # Called bar
    print(test.bar) # foo-bar
    print(test.bar) # foo-bar
    
    test.foo = "FOO"
    print()

    print(test.foo) # FOO
    # Called bar
    print(test.bar) # FOO-bar
    print(test.bar) # FOO-bar

    print(test.cache.getEntry("foo").cascade())