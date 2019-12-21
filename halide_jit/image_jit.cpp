#include <unistd.h>

#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;
using namespace Halide::ConciseCasts;
using namespace Halide::Tools;

int main(int argc, char **argv) {

    Halide::Buffer<uint8_t> image = load_image(argv[1]);

    Halide::Func input, blur;
    Halide::Var x, y, c;

	input(x, y, c) = cast<uint16_t>(BoundaryConditions::repeat_edge(image)(x, y, c));
#if 0
	// DO 3x3 blur HERE, currenly just bypass the image to output
    blur(x, y, c) = u8(input(x, y, c));
    std::cout << x << std::endl;
#elif 0
    blur(x, y, c) = u8((
                input(x - 1, y - 1, c) +
                input(x    , y - 1, c) +
                input(x + 1, y - 1, c) +
                input(x - 1, y    , c) +
                input(x    , y    , c) +
                input(x + 1, y    , c) +
                input(x - 1, y + 1, c) +
                input(x    , y + 1, c) +
                input(x + 1, y + 1, c)) / 9
            );
#else
    RDom r(-1, 3, -1, 3);
    blur(x, y, c) = u8(sum(input(x + r.x, y + r.y, c)) / 9);
#endif

    Halide::Buffer<uint8_t> output = blur.realize(image.width(), image.height(), image.channels());

    save_image(output, "blur.png");

    printf("Success!\n");
    return 0;
}
