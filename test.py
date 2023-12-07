import numpy as np

numbers = [
    103.69,
    104.11,
    104.76,
    180.3,
    180.85,
    256.8,
    257.73,
    329.97,
    331.71,
    401.87,
    402.21,
]


# remove numbers that are too close to each other
def remove_close_numbers(numbers):
    numbers = np.sort(numbers)
    new_numbers = []
    for i, number in enumerate(numbers):
        if i == 0:
            new_numbers.append(number)
        else:
            if abs(number - new_numbers[-1]) > 40:
                new_numbers.append(number)
    return new_numbers


print(remove_close_numbers(numbers))
