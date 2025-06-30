class SparseVector:
    def __init__(self):
        self.counts = {}
        self.key_set = set()
        self.norm = 0

    @staticmethod
    def _add_vectors(vec1, vec2):
        new_vector = SparseVector()
        key_set = vec1.key_set.union(vec2.key_set)
        for key in key_set:
            in_vec1 = key in vec1.key_set
            in_vec2 = key in vec2.key_set
            if in_vec1 and not in_vec2:
                new_vector[key] = vec1[key]
            elif not in_vec1 and in_vec2:
                new_vector[key] = vec2[key]
            elif in_vec1 and in_vec2:
                new_vector[key] = vec1[key] + vec2[key]
        return new_vector

    @staticmethod
    def _multiply_vectors(vec1, vec2):
        new_vector = SparseVector()
        for key in vec1.key_set:
            if key not in vec2.key_set:
                new_vector[key] = 0
            else:
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
            self.norm = self.norm**2 - self[key]**2
            self.counts[key] += value
            self.norm += self[key] ** 2
            self.norm = self.norm**0.5

    @staticmethod
    def dot(vec1, vec2):
        if len(vec1) < len(vec2):
            return SparseVector._dot(vec1, vec2)

        return SparseVector._dot(vec2, vec1)

    @staticmethod
    def _dot(vec1, vec2):
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
        if len(self) < len(other):
            return SparseVector._add_vectors(self, other)

        return SparseVector._add_vectors(other, self)

    def __mul__(self, other):
        if isinstance(other, SparseVector):
            if len(self) < len(other):
                return SparseVector._multiply_vectors(self, other)

            return SparseVector._multiply_vectors(other, self)
        return self._scalar_multiply(other)

    def __rmul__(self, other):
        if isinstance(other, SparseVector):
            if len(self) < len(other):
                return SparseVector._multiply_vectors(self, other)

            return SparseVector._multiply_vectors(other, self)
        return self._scalar_multiply(other)

    def __truediv__(self, other):
        return self._scalar_multiply(1 / other)

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError(f"Key must be an integer. Received invalid type: {type(key)}")
        self.counts[key] = value
        self.key_set.add(key)
        self.norm = (self.norm**2 + value**2)**0.5

    def __getitem__(self, item):
        return self.counts[item]

    def __iter__(self):
        for key in self.counts:
            yield key

    def __len__(self):
        return len(self.counts)
