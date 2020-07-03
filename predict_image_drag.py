# Custom imports
from dataset import *

### ************************************************
### Main execution
# Parameters
img_name    = 'imgs/NACA_6412_shape_croped.png'
img_width   = 256
img_height  = 256
downscaling = 2
drag_color  = 'bw'

# Load model
model_drag = load_h5_model('model.h5')

# Compute image size
height   = math.floor(img_height/downscaling)
width    = math.floor(img_width /downscaling)

# Predict drag
img      = load_and_reshape_img(img_name, height, width, 'bw')
img      = img.reshape(1,height,width,1)
drag     = model_drag.predict(img)

print('****************************')
print('PREDICTED DRAG :',drag)
print('****************************')
