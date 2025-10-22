from yopo import YOPO
from pathlib import Path

def main():
    modal = YOPO(
        modal_size='n',
    )

    image = Path('/home/batman/Projects/Y.O.P.O/training/dataset/images/val/creative_www.photopea.com_1729626995.png')

    results = modal.predict(source=image)

    print(results)


if __name__ == '__main__':
    main()
