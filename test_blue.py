from modules.petsnal_colors import *

def do_test():
    image = process_image_from_url("https://puddybuddybucket.s3.amazonaws.com/images/0af56026-d215-4fa6-b20b-900b227c8f98_1.jpeg")
    insert_image = Image.open("assets/clothes/76.png")

    image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pillow_image = Image.fromarray(image)
    # insert_image = insert_image.convert("RGBA")

    print(pillow_image.mode)
    print(insert_image.mode)

    # 이미지 합성
    pillow_image.paste(insert_image, (0,0), mask=insert_image)

    url = upload_to_s3(pillow_image)
    print(url)
    return



if __name__ == '__main__':
    do_test()
