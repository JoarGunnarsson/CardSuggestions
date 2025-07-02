class SparseVector:
    def __init__(self):
        self.counts = {}
        self.key_set = set()
        self.norm = 0

    @staticmethod
    def _multiply_vectors(vec1, vec2):
        new_vector = SparseVector()
        intersection = vec1.key_set.intersection(vec2.key_set)
        for key in intersection:
            new_vector[key] = vec1[key] * vec2[key]
        return new_vector

    def sum(self):
        total_sum = 0
        for key in self.key_set:
            total_sum += self.counts[key]
        return total_sum

    def increment(self, key, value=1):
        if not isinstance(key, int):
            raise TypeError(f"Key must be an integer. Received invalid type: {type(key)}")
        if key not in self.key_set:
            self[key] = value
        else:
            self[key] = self[key] + value

    @staticmethod
    def dot(vec1, vec2):
        dot_product = 0
        intersection = vec1.key_set.intersection(vec2.key_set)
        for key in intersection:
            dot_product += vec1[key] * vec2[key]
        return dot_product

    def _scalar_multiply(self, num):
        new_vector = SparseVector()
        for key in self.key_set:
            new_vector[key] = self.counts[key] * num
        return new_vector

    def __add__(self, other):
        new_vector = SparseVector()
        key_set = self.key_set.union(other.key_set)
        for key in key_set:
            new_vector[key] = self[key] + other[key]
        return new_vector

    def __mul__(self, other):
        if isinstance(other, SparseVector):
            return SparseVector._multiply_vectors(self, other)

        return self._scalar_multiply(other)

    def __rmul__(self, other):
        if isinstance(other, SparseVector):
            return SparseVector._multiply_vectors(self, other)

        return self._scalar_multiply(other)

    def __truediv__(self, other):
        return self._scalar_multiply(1 / other)

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError(f"Key must be an integer. Received invalid type: {type(key)}")

        if key in self.key_set:
            new_norm = self.norm ** 2 - self[key] ** 2
        else:
            new_norm = self.norm ** 2

        if value == 0:
            if key in self.key_set:
                self.key_set.remove(key)
                del self.counts[key]
            self.norm = new_norm ** 0.5
            return

        self.key_set.add(key)
        self.counts[key] = value
        self.norm = (new_norm + value**2)**0.5

    def __getitem__(self, item):
        if item in self.key_set:
            return self.counts[item]
        return 0

    def __iter__(self):
        for key in self.counts:
            yield key

    def __len__(self):
        return len(self.counts)
