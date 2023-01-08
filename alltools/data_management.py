from typing import Optional, Union
import csv


def append_row_to_csv(filename: str, row: list[str]) -> None:
    """Appends a row to a CSV file.

    Args:
        filename: The name of the CSV file.
        row: The row to be added to the file, as a list of values.
    """
    # Open the CSV file in append mode
    with open(filename, 'a', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        # Add the row to the CSV file
        writer.writerow(row)

def append_column_to_csv(filename: str, column: list[str]) -> None:
    """Appends a column to a CSV file.

    Args:
        filename: The name of the CSV file.
        column: The column to be added to the file, as a list of values. The value at index `i` will be added to the `i`-th row.
    """
    # Open the CSV file for reading
    with open(filename, 'r') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)
        # Read all the rows in the file
        rows = list(reader)

    # Add the new column to each row
    for i, row in enumerate(rows):
        row.append(column[i])

    # Open the CSV file in write mode
    with open(filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        # Write all the rows to the file
        writer.writerows(rows)


def dict2str(
    dictionary: dict,
    *,
    space: Optional[int] = 1,
    tabulation: Optional[int] = 2,
    first_space: Optional[bool] = True
) -> str:
    """Converts :obj: `dict` into readable `str`

    Args:
        dictionary (dict): A dictionary to convert
        space (:obj: `int`, optional) Number of spaces between depth levels. Defaults to 1.
        tabulation (:obj: `int`, optional): Number of spaces
            at the first level of the dictionary depth. Defaults to 2.
        first_space (:obj: `bool`, optional): To add spaces at the first level
            of the dictionary depth. Defaults to True.

    Returns:
        str: A converted dictionary
    """

    def tab(tabulation: int) -> str:
        return ' ' * tabulation

    string = f'{tab(tabulation) if first_space else ""}{{\n'
    tabulation += space
    for key, value in zip(dictionary.keys(), dictionary.values()):
        if not isinstance(key, str):
            key = f'{key}'
        if '\n' in key:
            key = key.replace('\n', '')
        string += f'{tab(tabulation)}{key}: '
        if not isinstance(value, dict):
            string += f'{value},\n'
        else:
            string += dict2str(value, space=space, tabulation=tabulation + space, first_space=False)
    string += f'{tab(tabulation - space)}}}\n'
    return string


def convert_base(
    num: Union[int, str],
    to_base: Optional[int] = 10,
    from_base: Optional[int] = 10
) -> str:
    """Converts base of a given number

    Args:
        num (:obj: `int` or :obj: `str`): Number to convert base.
        to_base (:obj: `int`, optional): The base to convert to. Defaults to 10.
        from_base (:obj: `int`, optional): The base to convert from. Defaults to 10.

    Raises:
        ValueError: If the given base is out of allowed range: [2, 36]

    Returns:
        str: A converted number.
    """
    if from_base < 2 or from_base > 36:
        raise ValueError('from_base must be >= 2 and <= 36, or 0')
    if to_base < 2 or to_base > 36:
        raise ValueError('to_base must be >= 2 and <= 36, or 0')
    if isinstance(num, str):
        n = int(num, from_base)
    else:
        n = int(str(num), from_base)
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n < to_base:
        return alphabet[n]
    else:
        return convert_base(n // to_base, to_base) + alphabet[n % to_base]
