# coding:utf-8
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt


def stretch_function_values(old_value, old_min=-math.exp(-5), old_max=math.exp(5), new_min=-100, new_max=100):

    scaled_value = (old_value - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min

    return scaled_value

def stretch_histogram(hist):
    """
    Calculate the gray histogram and stretch the non-zero pixel value to the range of 0-255

    Args:
    Image_path (STR): image path

    Returns:
    Tuple: stretched histogram, original histogram
    """

    # Find the minimum and maximum values of non-zero pixels
    min_nonzero = np.min(np.nonzero(hist)[0])
    max_nonzero = np.max(np.nonzero(hist)[0])

    #
    stretch_ratio = (255 - 0) / (max_nonzero - min_nonzero)

    #
    new_hist = np.zeros(256, dtype=np.float32)

    # Map non-zero pixel values to a new histogram
    for i in range(min_nonzero, max_nonzero + 1):
        new_hist[int((i - min_nonzero) * stretch_ratio)] = hist[i]

    return new_hist

def get_circle_brightness(image):

  # circle detection
  circles = detect_circles(image)

  # Extract circular region
  circle = circles[0][0]
  x, y, r = circle
  x,y,r=int(x),int(y),int(r)
  mask = np.zeros(image.shape[:2], dtype=np.uint8)
  cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
  roi = cv2.bitwise_and(image, image, mask=mask)
  #cv2.imwrite('roi.png',roi)
  #gray_normalized = cv2.normalize(mask, 0, 255, cv2.NORM_MINMAX)
  #
  #gray = cv2.cvtColor(roi)
  hist = cv2.calcHist([roi], [0], mask, [256], [0, 256])
  #cdf1 = np.cumsum(hist)
  stretched_hist = stretch_histogram(hist)
  #
  cdf = np.cumsum(stretched_hist)

  # Calculate the median position of the circular area
  median_pos = np.where(cdf >= cdf[-1] / 2)[0][0]
  print("median position：",median_pos)

  return median_pos/255


# circle
def detect_circles(image):
  #
  #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  #
  circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 2000,
                          param1=100, param2=30, minRadius=500, maxRadius=1024)


  if circles is None:
      #
      circles = [[[1024, 1024, 900]]]

    # Returns a list of circular areas
  #print(circles)
  return circles


def compute_mean_brightness(image):
    # Calculate the average brightness of each column of the image
    mean_brightness = np.mean(image, axis=0)
    return mean_brightness

def main():
    #parh
    image_path = './dataHa_name/test/cloud/'
    files_clear = os.listdir(image_path)
    eor=[]
    clean=[]
    cloud1=[]
    cloud2=[]
    cloud3=[]
    data=[]
    for j in range(len(files_clear)):
        img = cv2.imread(image_path+files_clear[j], cv2.IMREAD_GRAYSCALE)
        print(files_clear[j][:-4])
        brightness = get_circle_brightness(img)
        h, w = img.shape[:2]
        maxRadius = math.hypot(w / 2, h / 2)
        m = w / math.log(maxRadius)
        log_polar = cv2.logPolar(img, (w / 2, h / 2), m, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
        #log_polar = cv2.linearPolar(img, (w / 2, h / 2), maxRadius, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
        #log_polar = log_polar[:, 0:1500]

        # Divide the image into four parts
        height, width = log_polar.shape
        quarter_height = height // 4
        quarters = [log_polar[i:i + quarter_height, :] for i in range(0, height, quarter_height)]
        r = []
        a=0
        # Calculate the average brightness of each part
        mean_brightness_values = [compute_mean_brightness(quarter) for quarter in quarters]
        # draw curve
        for i, mean_brightness in enumerate(mean_brightness_values):
            if i<=2:
                correlation_coefficient = np.corrcoef(mean_brightness_values[i], mean_brightness_values[i+1])[0, 1]
                r.append(correlation_coefficient)
                # for jj, value in enumerate(mean_brightness_values[i]):
                #     if jj < 1800:
                #         if value <= 70:
                #             a += 1
            else:
                r.append(np.corrcoef(mean_brightness_values[3], mean_brightness_values[0])[0, 1])
                r.append(np.corrcoef(mean_brightness_values[0], mean_brightness_values[2])[0, 1])
                r.append(np.corrcoef(mean_brightness_values[1], mean_brightness_values[3])[0, 1])
                # for jj, value in enumerate(mean_brightness_values[i]):
                #     if jj < 1800:
                #         if value <= 70:
                #             a += 1
            #plt.plot(mean_brightness, label=f'Quarter {4 - i}')
        min = 1
        for i,v in enumerate(r):
            if r[i]<min:
                min=r[i]
            if i==len(r)-1:
                print("The image "+files_clear[j][:-4]+' min r: '+str(min))
                print("Median of normalized pixel intensity distribution："+str(brightness))
                c = math.log(min+2)* math.exp((brightness-0.5))
                data.append(c)
                print("The image "+files_clear[j][:-4]+' c: '+str(c))
                print("")
                if c>1.25:
                    clean.append(files_clear[j][:-4])
                elif  c>1.1:
                    cloud1.append(files_clear[j][:-4])
                elif  c>0.9:
                    cloud2.append(files_clear[j][:-4])
                else:
                    cloud3.append(files_clear[j][:-4])

    print(data)
    print("The pictures without cloud contamination are："+str(clean)+" total:"+str(len(clean)))
    print("Pictures of mild cloud contamination include：" + str(cloud1)+" total:"+str(len(cloud1)))
    print("Pictures of moderate cloud contamination include：" + str(cloud2)+" total:"+str(len(cloud2)))
    print("Pictures of severe cloud contamination include：" + str(cloud3)+" total:"+str(len(cloud3)))
    print("other：" + str(eor)+" total"+str(len(eor)))


if __name__ == "__main__":
    main()
