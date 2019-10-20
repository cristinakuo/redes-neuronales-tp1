from hopfield import *
import image
from utils import sum_patterns
from utils import invert_pattern
# Combine patterns and evaluate in trained net
def test_combination(neural_net, testing_set, render=False):
    log.info("Testing patterns...")
    L = len(testing_set)

    # Combine patterns
    combined_pattern = image.load_binary_image(testing_set[0])["data"]
    for i in range(1,L):
        next_pattern = image.load_binary_image(testing_set[i])["data"]
        # Load test pattern
        combined_pattern = sum_patterns(combined_pattern, next_pattern)
    image.render_image(combined_pattern, neural_net.rows, neural_net.cols)
    input("Press enter to continue...")
    
    # Evaluate
    refreshed = neural_net.evaluate_net(combined_pattern, 'async', 20,render=render)
    image.render_image(refreshed, neural_net.rows, neural_net.cols)
    input("Press enter to continue...")

def test_inverted(neural_net, testing_file, render=False):
    test_pattern = image.load_binary_image(testing_file)["data"]
    inverted_pattern = invert_pattern(test_pattern)
    image.render_image(inverted_pattern, neural_net.rows, neural_net.cols)
    input("Press enter to continue...")
    
    # Evaluate
    refreshed = neural_net.evaluate_net(inverted_pattern, 'async', 20,render=render)
    image.render_image(refreshed, neural_net.rows, neural_net.cols)
    input("Press enter to continue...")



def main():
    plt.ion()

    training_set = [
        "img/panda.bmp",
        "img/v.bmp",
        "img/perro.bmp"
    ]
    testing_set = [
        "img/panda.bmp",
        "img/perro.bmp",
        "img/v.bmp"
    ]

    myHop = HopfieldNet()
    myHop.load_image_patterns(training_set)
    myHop.train()

    test_combination(myHop, testing_set, render=True)
    
    #test_inverted(myHop, "img/perro.bmp", render=True)

if __name__ == '__main__':
    main()
