global volume

volume = None
alat = 10

def shift_volume():
    global volume
    compute_volume()
    volume = volume-3333

def compute_volume():
    global volume
    volume = alat**3

def print_volume():
    print(volume)
