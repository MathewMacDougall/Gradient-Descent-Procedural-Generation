from enum import IntEnum, unique, auto

# A custom value-generator for enums that returns increasing integers
# starting from 0
class DataEnumValueGenerator(IntEnum):
    def _generate_next_value_(name, start, count, last_values):
        return start + count - 1
