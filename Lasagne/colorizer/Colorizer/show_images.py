from Colorizer import Colorizer

def get_n_images():
    while True:
        try:
            n_images = int(input("How many images to show?"))       
        except ValueError:
            print("It should be an integer, please try again!")
            continue
        else:
            break

    return n_images

NNcolorizer = Colorizer(param_file='params_landscape.npy')

# Obtain settings from user
n_images = get_n_images()

# Now do untill the program closes:
while True:

    try:
        NNcolorizer.show_random_images_with_UV_channels(n_images)
    except:
        print("Something went wrong...")
    
    print("Do you want to evaluate more images?")
    n_images = get_n_images()

