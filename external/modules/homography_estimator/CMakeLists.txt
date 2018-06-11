set(SOURCES
    "src/homography_estimator.cpp"
    "src/interface.cpp"
)

set(HEADERS
    "include/homography_estimator.h"
    "include/homography.h"
)

find_package( OpenCV REQUIRED )

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)
add_library(homography_estimator MODULE ${SOURCES} ${HEADERS})
target_link_libraries(homography_estimator PRIVATE lmscore imaging ${OpenCV_LIBS})
