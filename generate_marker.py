import cv2
import cv2.aruco as aruco
import os

def generate_marker(marker_id=0, size_px=400, output_dir="data"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # We use DICT_4X4_50 for simple, clear, large blocks (good for screens)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    # Generate the marker image
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    
    # Add a mandatory white border (quiet zone) so it works on dark-mode phones!
    border_size = int(size_px * 0.15)
    marker_image = cv2.copyMakeBorder(
        marker_image, 
        border_size, border_size, border_size, border_size, 
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    
    # Save the image
    filename = os.path.join(output_dir, f"aruco_marker_id_{marker_id}.png")
    cv2.imwrite(filename, marker_image)
    print(f"Generated ArUco Marker ID {marker_id} at {filename}")
    print("You can open this image and display it full screen on your phone or tablet!")

if __name__ == "__main__":
    generate_marker(0)
