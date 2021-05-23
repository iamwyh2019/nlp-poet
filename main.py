from utils.dataloader import Poet_Dataset

def main():
    a = Poet_Dataset("data/wuyanjueju.txt")
    a.Test()

if __name__ == "__main__":
    main()