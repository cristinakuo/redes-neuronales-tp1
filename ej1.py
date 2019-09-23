
from hopfield import *
import image

def test(neural_net, testing_set, render=False):
    log.info("Testing patterns...")
    
    for test in testing_set:
        # Load test pattern
        test_pattern = image.load_binary_image(test)["data"]
        
        # Add noise to test pattern
        test_pattern_noisy = add_noise(test_pattern, 0.25)
        image.render_image(test_pattern_noisy, neural_net.rows, neural_net.cols)
        input("Press enter to continue...")
        
        refreshed = neural_net.evaluate_net(test_pattern_noisy, render)
        
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
        "img/panda.bmp"
    ]

    myHop = HopfieldNet()
    myHop.load_image_patterns(training_set)
    myHop.train()
    
    test(myHop, testing_set, render=True)

if __name__ == '__main__':
    main()