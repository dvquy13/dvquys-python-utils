import re

viettel = "^(0|84)*(96|97|98|162|163|164|165|166|167|168|169|86)\d{7}$"
def is_viettel(isdn):
    isdn = str(isdn)
    match = re.search(viettel, isdn)
    if match:
        return True
    else:
        return False

assert(is_viettel(1683130038))
assert(is_viettel("1683130038"))
assert(~is_viettel("168313003a"))
assert(is_viettel("01683130038"))
assert(is_viettel("841683130038"))
