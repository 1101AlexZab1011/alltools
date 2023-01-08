from functools import wraps
import hashlib
from typing import Optional, Callable, Any, Union, Dict
import os
import configparser
from functools import wraps


def read_config_from_file(filename: str):
    # Create a ConfigParser instance
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(f'{filename}.cfg')

    # Return the config as a Python object
    return config


def check_path(*args: str) -> None:
    """Creates a directory if does not exist and if it is possible.

    Args:
        *args (str): Directories to check.
    """
    for path in args:
        if not os.path.isdir(path):
            os.mkdir(path)


def get_dir_size(start_path: str = './') -> float:
    """Computes size of directory in bites

    Args:
        start_path (str, optional): Path to compute size of which. Defaults to './'.

    Returns:
        float: Size of the given directory in bites
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def read_or_write(
        file_ext: str,
        reader: Optional[Callable] = None,
        writer: Optional[Callable] = None,
        path: Optional[Union[str, Callable]] = './rw_storage',
        file_prefix: Optional[Union[str, Callable]] = ''
):
    """A decorator for calculating a decorated function if it has never been executed
        and save a result or to read the result without execution
        of the function if it was computed before.

    Args:
        file_ext (str): An extension of the file to save result of the decorated function
        reader (:obj: `Callable`, optional): Function to read a result if it already exists.
            Must take path to the stored content to read as the single argument.
            If None, does not read. Defaults to None.
        writer (:obj: `Callable`, optional): Function to write a result.
            Must take path as the first argument and any content to save as the second.
            If None, does not write. Defaults to None.
        path (:obj: `str` or :obj: `Callable`, optional): Path to storage for results.
            Defaults to './rw_storage'.
        file_prefix (:obj: `str` or :obj: `Callable`, optional): Prefix to savefile
            to make it more human-readable. Defaults to ''.
    """
    def encrypt_callable(func: Callable, args: tuple, kwargs: dict[str: Any], prefix: str):
        args_str = ", ".join([str(arg) for arg in args])
        kwargs_str = ', '.join([f'{key}={value}' for key, value in kwargs.items()])
        params_str = args_str + ', ' + kwargs_str if not not args_str else kwargs_str
        return prefix + str(hashlib.md5(
            bytes(
                f'{func.__name__}({params_str})',
                'utf-8'
            )
        ).hexdigest())

    def write(func: Callable, args: tuple, kwargs: Dict[str, Any], path: str, prefix: str) -> None:
        result = func(*args, **kwargs)
        if writer is not None:
            check_path(path)
            writer(
                os.path.join(
                    path,
                    f'{encrypt_callable(func, args, kwargs, prefix)}.{file_ext}'
                ),
                result
            )
        return result

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:

            if isinstance(path, Callable):
                place = path(*args, **kwargs)
            elif isinstance(path, str):
                place = path
            else:
                raise ValueError(f'Path have to be string or callable, but {type(path)} is given')

            check_path(path)

            if isinstance(file_prefix, Callable):
                prefix = file_prefix(*args, **kwargs)
            elif isinstance(file_prefix, str):
                prefix = file_prefix
            else:
                raise ValueError(
                    f'Prefix to file have to be string or callable, but {type(path)} is given'
                )

            if reader is not None and os.path.exists(place):
                code = encrypt_callable(func, args, kwargs, prefix)
                for address, _, files in os.walk(place):
                    for file in files:
                        if code in file:
                            return reader(os.path.join(address, file))
            return write(func, args, kwargs, place, prefix)

        return wrapper

    return decorator


def read_config(directory: str = './') -> Callable[[Callable], Callable]:
    """
    A decorator that reads configuration from a file with the same name as the decorated function.

    The configuration file is assumed to be in the '.cfg' format and located in the specified `directory`.
    If no arguments or keyword arguments are provided when calling the decorated function, the parameters
    from the configuration file are used instead.

    Args:
        directory: The directory where the configuration file is located. Default: './' (the current directory)

    Returns:
        A decorator that can be applied to a function to read its configuration from a file.
    """
    # [DEFAULT]
    # args = 1, 2
    # kwargs = kwarg1=foo, kwarg2=bar
    def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Read the configuration file with the same name as the function, located in the specified directory
                config = read_config_from_file(f'{directory}/{func.__name__}.cfg')

                # If no arguments or keyword arguments are provided, use the parameters from the config file
                if not args and not kwargs:
                        args = config.args
                        kwargs = config.kwargs

                # Call the decorated function with the updated arguments and keyword arguments
                return func(*args, **kwargs)

            return wrapper

    return decorator
