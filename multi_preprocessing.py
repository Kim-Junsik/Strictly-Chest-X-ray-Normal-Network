import torch
import numpy as np
import random
import cv2

class ToTensor(object):
    def __call__(self, data):
        if 'recon' in data.keys():
            input, recon, mask, label = data['image'], data['recon'], data['mask'], data['label']
            input = input.transpose((2, 0, 1)).astype(np.float32)
            recon = recon.transpose((2, 0, 1)).astype(np.float32)
            mask = mask.transpose((2, 0, 1)).astype(np.float32)
            
            data = {'image': torch.from_numpy(input), 'recon': torch.from_numpy(recon), 'mask': torch.from_numpy(mask), 'label': label}

            return data

        input, mask, label = data['image'], data['mask'], data['label']
        input = input.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        
        data = {'image': torch.from_numpy(input), 'mask': torch.from_numpy(mask), 'label': label}

        return data

class Gamma_2D(object):
    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        image = image.astype("uint8")
        
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def __call__(self, sample):
        p = 0.3
        if 'recon' in sample.keys():
            image = sample['image']
            recon = sample['recon']
            mask = sample['mask']
            label = sample['label']
            
            if random.random() < p:
                var_gamma = random.uniform(0.8, 1.5)
                if not isinstance(var_gamma, float):
                    var_gamma = var_gamma[0]
                image = self.adjust_gamma(image, gamma=var_gamma)
                image = image.astype("uint8")
                image = np.expand_dims(image, axis=-1)

                recon = self.adjust_gamma(recon, gamma=var_gamma)
                recon = recon.astype("uint8")
                recon = np.expand_dims(recon, axis=-1)
            return {'image': image, 'recon': recon, 'mask': mask, 'label': label}
        
        
        image = sample['image']
        mask = sample['mask']
        label = sample['label']
        
        if random.random() < p:
            var_gamma = random.uniform(0.8, 1.5)
            if not isinstance(var_gamma, float):
                var_gamma = var_gamma[0]
            image = self.adjust_gamma(image, gamma=var_gamma)
            image = image.astype("uint8")
            image = np.expand_dims(image, axis=-1)
        return {'image': image, 'mask': mask, 'label': label}


class Rotation_2D(object):
    def __call__(self, sample, degree = 10):
        p = 0.3
        if 'recon' in sample.keys():
            image = sample['image']
            recon = sample['recon']
            mask = sample['mask']
            label = sample['label']
            R_move = random.randint(-degree,degree)
            if random.random() < p:
                #print("_rotation_2D")
                M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), R_move, 1)
                image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
                image = np.expand_dims(image, axis=-1)
                recon = cv2.warpAffine(recon,M,(recon.shape[1],recon.shape[0]))
                recon = np.expand_dims(recon, axis=-1)
                for i in range(mask.shape[2]):
                    mask[:,:,i] = cv2.warpAffine(mask[:,:,i],M,(image.shape[1],image.shape[0]))
                # mask = np.expand_dims(mask, axis=-1)
                #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
            return {'image': image, 'recon': recon, 'mask': mask, 'label': label}

        image = sample['image']
        mask = sample['mask']
        label = sample['label']
        R_move = random.randint(-degree,degree)
        if random.random() < p:
            #print("_rotation_2D")
            M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), R_move, 1)
            image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
            image = np.expand_dims(image, axis=-1)
            for i in range(mask.shape[2]):
                mask[:,:,i] = cv2.warpAffine(mask[:,:,i],M,(image.shape[1],image.shape[0]))
            # mask = np.expand_dims(mask, axis=-1)
            #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
        return {'image': image, 'mask': mask, 'label': label}


class Shift_2D(object):
    def __call__(self, sample, shift = 10):
        p = 0.3
        if 'recon' in sample.keys():
            image = sample['image']
            recon = sample['recon']
            mask = sample['mask']
            label = sample['label']
            
            x_move = random.randint(-shift,shift)
            y_move = random.randint(-shift,shift)
            if random.random() < p:
                shift_M = np.float32([[1,0,x_move], [0,1,y_move]])
                image = cv2.warpAffine(image, shift_M,(image.shape[1], image.shape[0]))
                recon = cv2.warpAffine(recon, shift_M,(recon.shape[1], recon.shape[0]))
                for i in range(mask.shape[2]):
                    mask[:,:,i] = cv2.warpAffine(mask[:,:,i], shift_M, (mask.shape[1], mask.shape[0]))
                image = np.expand_dims(image, axis=-1)
                recon = np.expand_dims(recon, axis=-1)
                
            return {'image': image, 'recon': recon, 'mask': mask, 'label': label}

        image = sample['image']
        mask = sample['mask']
        label = sample['label']
        
        x_move = random.randint(-shift,shift)
        y_move = random.randint(-shift,shift)
        if random.random() < p:
            shift_M = np.float32([[1,0,x_move], [0,1,y_move]])
            image = cv2.warpAffine(image, shift_M,(image.shape[1], image.shape[0]))
            for i in range(mask.shape[2]):
                mask[:,:,i] = cv2.warpAffine(mask[:,:,i], shift_M, (mask.shape[1], mask.shape[0]))
            image = np.expand_dims(image, axis=-1)
#             mask = np.expand_dims(mask, axis=-1)
            
        return {'image': image, 'mask': mask, 'label': label}


class RandomSharp(object):
    def __call__(self, sample):
        image = sample['image']
        p = 0.7 #0.3
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
        if random.random() < p:
            
            image = cv2.filter2D(image, -1, kernel)
            image = np.expand_dims(image, axis=-1)
            sample['image'] = image
                   
        return sample


class RandomBlur(object):
    def __call__(self, sample):
        image = sample['image']
        p = 0.6 #0.3
        if random.random() < p:
            image = sample['image']
            image = cv2.blur(image,(3,3))
            image = np.expand_dims(image, axis=-1)
            sample['image'] = image

        return sample


class RandomNoise(object):
    def __call__(self, sample):
        image = sample['image']
        
        p = 0.7 #0.3
        if random.random() < p:
            image = image/255.0
            noise =  np.random.normal(loc=0, scale=1, size=image.shape)
            img2 = image*2
            n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.05)), (1-img2+1)*(1 + noise*0.05)*-1 + 2)/2, 0,1)
            n2 = n2 * 255
            n2 = n2.astype("uint8")
            sample['image'] = n2         
        
        return sample

####################################################################################################################################################

class RandomFlip(object):
    def __call__(self, sample):
        p = 0.3
        image = sample['image']
        mask = sample['mask']
        label = sample['label']
        
        if random.random() < p:
            image = cv2.flip(image, 0)
            image = np.array(image)
            image = np.expand_dims(image, axis=2)
            print(mask.shape)
            raise
            for i in range(mask.shape[2]):
                mask[:,:,i] = cv2.flip(mask[:,:,i], 0)
        return {'image': image, 'mask': mask, 'label': label}


class Randomcrop(object):
    def __init__(self, scale=0.95, mode='seg'):
        assert scale >= 0.0
        assert scale <= 1.0
        self.scale = scale
        self.mode = mode
        
    def __call__(self, sample):
        # segmentation and MTL
        if self.mode == 'seg':
            if 'recon' in sample.keys():
                img = sample['image']
                recon = sample['recon']
                mask = sample['mask']
                label = sample['label']
                # Crop image
                height, width = int(img.shape[0]*self.scale), int(img.shape[1]*self.scale)
                x = random.randint(0, img.shape[1] - int(width))
                y = random.randint(0, img.shape[0] - int(height))

                cropped_img = img[y:y+height, x:x+width]
                cropped_recon = recon[y:y+height, x:x+width]
                cropped_mask = mask[y:y+height, x:x+width]

                resized_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]))
                resized_recon = cv2.resize(cropped_recon, (recon.shape[1], recon.shape[0]))
                resized_mask = cv2.resize(cropped_mask, (mask.shape[1], mask.shape[0]))
                
                if len(resized_img.shape) == 2:
                    resized_img = np.expand_dims(resized_img,axis=-1)
                if len(resized_recon.shape) == 2:
                    resized_recon = np.expand_dims(resized_recon,axis=-1)
                if len(resized_mask.shape) == 2:
                    resized_mask = np.expand_dims(resized_mask,axis=-1)
                    
                new_sample = {'image':resized_img, 'recon': resized_recon, 'mask': resized_mask, 'label': label}
                
                return new_sample

            img = sample['image']
            mask = sample['mask']
            label = sample['label']
            # Crop image
            height, width = int(img.shape[0]*self.scale), int(img.shape[1]*self.scale)
            x = random.randint(0, img.shape[1] - int(width))
            y = random.randint(0, img.shape[0] - int(height))

            cropped_img = img[y:y+height, x:x+width]
            cropped_mask = mask[y:y+height, x:x+width]

            resized_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]))
            resized_mask = cv2.resize(cropped_mask, (mask.shape[1], mask.shape[0]))
            
            if len(resized_img.shape) == 2:
                resized_img = np.expand_dims(resized_img,axis=-1)

            if len(resized_mask.shape) == 2:
                resized_mask = np.expand_dims(resized_mask,axis=-1)

            new_sample = {'image':resized_img, 'mask': resized_mask, 'label': label}
            
            return new_sample
        # detection
        else:
            img = sample[0]
            target = sample[1]
            height, width = int(img.shape[0]*self.scale), int(img.shape[1]*self.scale)
            x = random.randint(0, img.shape[1] - int(width))
            y = random.randint(0, img.shape[0] - int(height))
            cropped_img = img[y:y+height, x:x+width]
            resized_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]))
            # Modify annotation
            new_boxes=[]
            for box in target['boxes']:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                x1, x2 = x1-x, x2-x
                y1, y2 = y1-y, y2-y
                x1, y1, x2, y2 = x1/self.scale, y1/self.scale, x2/self.scale, y2/self.scale
                if (x1<img.shape[1] and y1<img.shape[0]) and (x2>0 and y2>0):
                    if x1<0: x1=0
                    if y1<0: y1=0
                    if x2>img.shape[1]: x2=img.shape[1]
                    if y2>img.shape[0]: y2=img.shape[0]
                    new_boxes.append([x1, y1, x2, y2])
            new_target = target
            new_target['boxes'] = new_boxes
            return resized_img, new_target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, sample):
        img = sample['image']
        if img.shape[2] == 1:
            gray = True
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        delta = np.array([random.uniform(-self.delta, self.delta)]).astype('uint8')
        lim = 255 - delta
        v[v > lim] = 255
        v[v <= lim] += delta
        final_hsv = cv2.merge((h, s, v))
        if gray:
            v = np.expand_dims(v, axis=-1)
            sample['image'] = v
            return sample
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        sample['image'] = img
        return sample


class Noisy(object):
    def __init__(self, noise_type="gauss"):
        self.noise_type = noise_type

    def __call__(self, image):
        if self.noise_type == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif self.noise_type == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0
            return out
        elif self.noise_type == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif self.noise_type =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy