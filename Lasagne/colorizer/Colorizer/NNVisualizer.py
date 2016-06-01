
def CIELab_array2img(CIELab):
    """ INPUT:
                CIELab: Lab layers in an array of shape=(3,image_x, image_y)
        OUTPUT:
                Image
    """
    # Convert it to show the image!
    U_out = NN_output[0,0,:,:]*256
    V_out = NN_output[0,1,:,:]*256
    img_out = np.zeros((128,128,3), 'uint8')
    img_out[..., 0] = NN_input*256
    img_out[..., 1] = U_out
    img_out[..., 2] = V_out
    return Image.fromarray(img_out,'YCbCr')

def split_batch(batch):
    """ INPUT:
                batch: batch to split into input and target

        OUTPUT:
                a list containing the batch_input and the batch_target: [batch_input, batch_target]
    """

    # target is the UV layers
    batch_target = batch[:,[1,2],:,:]
    batch_target = batch_target.reshape(batch_size,2,image_x,image_y)
    # input is the Y layer
    batch_input = batch[:,0,:,:]
    batch_input = batch_input.reshape(batch_size,1,image_x,image_y)

    return [batch_input, batch_target]

def batch2img(batch, colorspace):
    """ INPUT: 
                batch: batch to convert to images
                colorspace:  the colorspace to convert from, either: 'CIELab' or 'RGB'

        OUTPUT: 
                list of images
    """

    (n_images, _, image_x, image_y) = batch.shape

    # Loop over the batch
    for img_number in range(n_images):


        # Get the original image:
        ORG_img = self.NNout2img(img_input,img_target)
        # Eval the NN
        NN_output = self.eval_fn(img_input)
        # Get NN image
        NN_img = self.NNout2img(img_input,NN_output)

    return [ORG_img, NN_img]

def show_random_images_with_UV_channels(n_images, folder='validation'):
    pass

def show_images_with_UV_channels(batch):
    """ INPUT:
                batch: batch of images with shape=(batch_size, 3, image_x, image_y)
    """
    n_images,_,_,_ = images.shape

    # Create figure
    f, ax = plot.subplots(n_images*2,3)

    # and loop over the images
    for index in range(0,n_images*2,2):

        # Get the image
        (ORG_img, NN_img) = self.batch2img(int(batch_ids[int(index/2)]),int(img_ids[int(index/2)]),folder=folder)
        # Get the U and V layers
        _, ORG_U, ORG_V = ORG_img.split()
        _, NN_U, NN_V = NN_img.split()

        # Show original image
        ax[index,0].axis('off')
        ax[index,0].imshow(ORG_img)
        # show the U layer
        ax[index,1].axis('off')
        ax[index,1].imshow(ORG_U,cmap='gray')
        # Show the V layer
        ax[index,2].axis('off')
        ax[index,2].imshow(ORG_V,cmap='gray')

        # Show the NN image
        ax[index+1,0].axis('off')
        ax[index+1,0].imshow(NN_img)
        # show the U layer
        ax[index+1,1].axis('off')
        ax[index+1,1].imshow(NN_U,cmap='gray')
        # Show the V layer
        ax[index+1,2].axis('off')
        ax[index+1,2].imshow(NN_V,cmap='gray')

    # Show the figures
    plot.show()