"""Enable the manupulation of loop progress output without passing variables"""
from tqdm import tqdm

report = tqdm

def use_tk():
    from tqdm.tk import tqdm
    global report
    report = tqdm

def use_std():
    global report, tqdm
    report = tqdm

def use_no_report():
    global report
    report = no_report

def no_report(iterator, **tqdm_flags):
    """Pass down iterator and ignore tqdm flags"""
    return iterator