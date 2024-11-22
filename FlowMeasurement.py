import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

#Create a Spreadsheet to save flow results
#Flow results using com denote results using a Center of Mass technique
#Flow results using rec denote results using a minimum enclosing rectangle.
with open('FlowMeasurement.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Vcom", "Vrec", "Qcom", "Qrec"])

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture("C:\\Users\\Jaden\\Desktop\\mainflow.mp4")

    # Create a Video Writer to save frames as a video
    fourcc2 = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('C:\\Users\\Jaden\\Desktop\\FlowMeasureVideo.mp4', fourcc2, 30, (720, 1280))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    #Booleans used to store images and filter results
    previousImage = False
    debugger = False
    passable = True
    data = []
    reached = False

#Variable to change when working with videos
#-------------------------------------------------------------------------------------------------------
    distanceCalib = 0.001/4 #0.001 meters per 4 pixels using ImageJ (convert videos from mp4 to avi and then avi to special avi using (in command line) ffmpeg -i inputfile.avi -pix_fmt nv12 -f avi -vcodec rawvideo outfile.avi

    #Manual settings to isolate tube using a rectangle, must use when lots of noise
    pipeWidth = 35 #x-axis width
    pipeStart = 205 #x-axis location (leftmost)
    pipeHeight = 1280 #This will stay the same since y will be from 0 to 1280
#--------------------------------------------------------------------------------------------------------

    totalFrames = 0 #Frame Counter

    #Result Storage arrays used for graphing purposes
    frimes = [] #Stores frame number
    flCom = [] #Stores flow rate using COM technique
    flRec = [] #Stores flow rate using min enclosing rectangle technique

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            #Increment frame counter
            totalFrames = totalFrames + 1

            #Convert image to greyscale
            img = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

            #Create seperate blank image
            blank = np.zeros(img.shape, "uint8")

            #Convert image to HSV (Hue, Saturation, and Value) instead of BGR - makes it easier to identify blue colour
            hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)

# These HSV values should be changed to create a mask where only the blue liquid is
# Acts as a range of HSV values that will be used to create a mask.
# Hue should be changed slightly to align with the blue color if needed.
# The lower saturation can be changed to improve velocity measurements (ensure both techniques provide similar results)
#------------------------------------------------------------------
            lower_blue = np.array([100, 100, 20])
            upper_blue = np.array([110, 255, 255])
#------------------------------------------------------------------

            # Here we are defining range of bluecolor in HSV
            # This creates a mask of blue coloured
            # objects found in the frame.
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # The bitwise and of the frame and mask is done so
            # that only the blue coloured objects are highlighted
            # and stored in res
            res = cv2.bitwise_and(frame, frame, mask=mask)

            # Turn the isolate blue components of the image into Grayscale
            img = cv2.cvtColor(res.copy(), cv2.COLOR_BGR2GRAY)


            # Check if there is a previous image to use for flow rate computation
            if (previousImage):

                # Used for manually fitting a rectangle around the pipe (use final in imshow manually adjust)
                blink = cv2.rectangle(blank.copy(), (pipeStart, 0), (pipeStart+pipeWidth, pipeHeight), (255, 255, 255), -1)
                final = img*(blink/255)

                # Apply a Morphological Open (erosion and then dilation) to remove noise
                kernel = np.ones((5,5), "uint8")
                final = cv2.morphologyEx(final.copy(), cv2.MORPH_OPEN, kernel)

                #Threshold the image into a binary image where the blue liquid is white
                ret, final = cv2.threshold(final.copy(), 10, 255, cv2.THRESH_BINARY)
                final = final.astype("uint8")

                #Find Contours in the threholded image to find the liquid
                contours, _ = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                area = [] #Store contour areas for filtering
                rects = [] #Store min enclosing (filtered) rectangle parameters and COM x and y
                chosenContours = [] #Stores the chosen contours (after area filtering)
                for i in range(len(contours)):
                    cnt = contours[i]

                    #Compute and store contour areas
                    ar = cv2.contourArea(cnt)
                    area.append(ar)

                    # Filter and store contour areas
                    if (ar > 2000):
                        chosenContours.append(cnt)

                        # Fit min enclosing rectangle for the chosen contours
                        (xa, ya, wa, ha) = cv2.boundingRect(cnt)

                        # Calculate moments of each chosen contour
                        M = cv2.moments(cnt)
                        x = int(M["m10"] / M["m00"])
                        y = int(M["m01"] / M["m00"])

                        # Store moments and rectangle parameters
                        rects.append([xa, ya, wa, ha, x, y])

                # Copy the frame image (for video)
                image1 = frame.copy()

                # Implement to draw all contours
                #cv2.drawContours(img.copy(), chosenContours, -1, (255, 255, 255), 3)

                # Draw the min enclosing rectangles and the center of mass of each contour (using circle) for video
                for p in range(len(rects)):
                    image1 = cv2.rectangle(image1.copy(), (rects[p][0], rects[p][1]), (rects[p][0]+rects[p][2], rects[p][1]+rects[p][3]), (255, 255, 255), 1)
                    image1 = cv2.circle(image1.copy(), (rects[p][4], rects[p][5]), 5, (255, 255, 255), -1)

                # Draw the manually set rectangle the encloses the pipe
                image1 = cv2.rectangle(image1.copy(), (pipeStart, 0), (pipeStart+pipeWidth, pipeHeight), (255, 255, 255), 1)

                # Check if the liquid has reached the top of the image using the top most point of the min enclosing boxes
                if (not(reached)):
                    for st in rects:
                        if (st[1] < 15):
                            reached = True

                # If the number of contours are the same as the previous frame, and there are contours, calculate flow rates
                # data stores the information (rects) of the previous frame
                # rects stores the information of the current frame
                # information of each contour is stored as [leftmost rect x, topmost rect y, rect width, rect hegiht, COM x, COM y]
                if (len(data) == len(rects) and len(data) != 0):
                    data = np.asarray(data)
                    rects = np.asarray(rects)

                    # Sort data (ascending) based on topmost rect y column (height of liquid)
                    data = data[data[:, 1].argsort()]
                    rects = rects[rects[:, 1].argsort()]

                    # Calculate the pixel distance the water moved by subtracting arrays
                    results = data - rects

                    # Ensure that the contours correspond to each other and are not staggered (ie. a different contour is closer to previous liquid position than the first contour in the rect, current position, list
                    if (len(data) > 1):
                        if (data[0][1]-rects[1][1] >= 0):
                            passable = False

                    # Additionally check that the liquid has not yet reached the top before calculating flow rates
                    if (not(reached) and passable):

                        # Calculate linear spead (m/s) and flow rate using min enclosing rectangle top edge difference (difference in y)
                        distanceRect = results[0][1]
                        linspeedRect = distanceCalib*distanceRect*480
                        FlowRect = linspeedRect*np.pi*0.00635*0.00635*1000*60/4

                        # Calculate linear spead (m/s) and flow rate using COM y difference, assuming uniform liquid area in contour in the form of a rectangle so a multiple of 2 is needed due to averaging of COM
                        distanceCOM = distanceCalib*2*(results[0][-1])
                        linspeedCOM = distanceCOM*480
                        FlowCOM = linspeedCOM*np.pi*0.00635*0.00635*1000*60/4

                        # Ensure Flow rates calculated are both positive
                        if(FlowRect > 0 and FlowCOM > 0):

                            # Print results
                            print("Results ", linspeedCOM, linspeedRect, FlowCOM, FlowRect)
                            print("data", data)
                            print("rects", rects)
                            print("results", results)

                            # Store results in spreadsheet
                            writer.writerow([totalFrames, linspeedCOM, linspeedRect, FlowCOM, FlowRect])

                            # Store results in corresponding storage arrays for plotting
                            frimes.append(totalFrames)
                            flCom.append(FlowCOM)
                            flRec.append(FlowRect)

                            # Use to troubleshoot sudden large value of flow rate
                            # if(FlowCOM > 30):#0.911 and FlowRect < 0.913):
                            #     debugger = True
                            #     while (True):
                            #         print("Np")


                # Store rects (contour information) to be used a s previous frame for next frame
                data = rects.copy()

                # Reset contour staggering check
                passable = True

                # Display the frame along with the min enclosing rectangles and COM of each contour
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 600, 600)
                cv2.imshow('image', image1) # final for manual fitting of rectangle around pipe

                # Wrtie the frame with the min enclosing rectangles and COM of each contour to a video
                out.write(image1)

            # There is now a previous image that cna possibly be used for flow rate calculations
            previousImage = True

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    out.release()


# Plot the flow rates for the two techniques against the frames - can be used for tuning of hsv range
plt.plot(frimes, flRec, label = "MinRec")
plt.plot(frimes, flCom, label = "COM")
plt.xlabel('Frames')
plt.ylabel('Flow Rate (LPM)')
plt.title('Flow Rate using Liquid Air Interface')
plt.legend()
plt.show()