import cv2
import os



def remove_hair(images_directory: str, new_directory: str) -> None:

    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    
    i = 0
    
    for img in os.listdir(images_directory):
        src = cv2.imread(images_directory+img)
        grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )
        kernel = cv2.getStructuringElement(1,(17,17))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
        dst = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA)
        cv2.imwrite(new_directory+img, dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        i+=1
        if i > 10:
            return


if __name__ == '__main__':

    image_paths = ["data/HAM10000_images_part_1/","data/HAM10000_images_part_2/"]
    new_directory = "data/images_hair_removed/"

    for path in image_paths:
        remove_hair(path,new_directory)