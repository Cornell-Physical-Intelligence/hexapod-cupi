# ROS 2 workspace

This directory holds the project's ROS 2 code. ROS code lives only here and may
import the repo's Python packages; nothing under the top-level `src/` should import
ROS, as the `uv` tests and CI run without a ROS install. Issue
[#103](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/103)
tracks the work.

| Path | Contents |
|---|---|
| `hexapod.repos` | Pinned revisions of the third-party sources |
| `patches/` | Changes applied to third-party sources after download |
| `src/hexapod_bringup/` | Launch files and parameter files |
| `src/external/` | Third-party sources, downloaded and git-ignored |

## Third-party sources

| Source | What it does |
|---|---|
| [Livox SDK](https://github.com/Livox-SDK/Livox-SDK2) | Library that talks to the Mid-360 lidar |
| [Livox ROS driver](https://github.com/Livox-SDK/livox_ros_driver2) | Publishes the lidar's data as ROS messages and defines their type |
| [FAST-LIO](https://github.com/hku-mars/FAST_LIO/tree/ROS2) | Computes the sensor's pose from lidar and motion-sensor data |
## Build

You need Ubuntu 24.04 with ROS 2 Jazzy (`ros-jazzy-desktop`) and
`ros-dev-tools`, which provides `colcon` and `vcs`. On Windows, use Ubuntu 24.04
in WSL 2 and keep the clone inside the Linux filesystem. Run every command from
`ros2_ws/`.

1. Install the two system libraries that the Jazzy desktop install lacks.

   ```sh
   sudo apt install ros-jazzy-pcl-ros libapr1-dev
   ```

2. Download the third-party sources at their pinned revisions. `--recursive`
   also fetches FAST-LIO's `ikd-Tree` submodule.

   ```sh
   mkdir -p src/external
   vcs import --recursive src/external < hexapod.repos
   ```

3. Build and install the Livox SDK. It is a plain CMake project, not a ROS
   package, and the driver looks for it in `/usr/local`. The `COLCON_IGNORE`
   file stops `colcon` from building it a second time in step 6.

   ```sh
   touch src/external/Livox-SDK2/COLCON_IGNORE
   cmake -S src/external/Livox-SDK2 -B src/external/Livox-SDK2/build -DCMAKE_BUILD_TYPE=Release
   cmake --build src/external/Livox-SDK2/build -j8
   sudo cmake --install src/external/Livox-SDK2/build
   sudo ldconfig
   ```

4. Prepare the driver. Its repository ships ROS 1 and ROS 2 variants of two
   items, and `colcon` needs the ROS 2 ones under the standard names. The
   driver's own `build.sh` does this too, but it also deletes `build/` and
   `install/` two directories above itself, so do not run it here.

   ```sh
   cp src/external/livox_ros_driver2/package_ROS2.xml src/external/livox_ros_driver2/package.xml
   cp -r src/external/livox_ros_driver2/launch_ROS2 src/external/livox_ros_driver2/launch
   ```

5. Patch FAST-LIO. Its build file requests the C++14 language standard, and
   Jazzy's ROS headers need C++17. The patch edits only `CMakeLists.txt`, the
   file that tells the compiler how to build the package, and changes only the
   language-standard setting there. It changes no source file, so the
   pose-estimation code is the code the Jetson ran. It is the change the author
   of FAST-LIO's ROS 2 port
   [recommends for Jazzy](https://github.com/Ericsii/FAST_LIO_ROS2/issues/28);
   upstream has not applied it
   ([FAST_LIO issue 400](https://github.com/hku-mars/FAST_LIO/issues/400)).

   ```sh
   git -C src/external/FAST_LIO apply ../../../patches/fast_lio_cxx17.patch
   ```

6. Build the workspace and load it into the shell.

   ```sh
   source /opt/ros/jazzy/setup.bash
   colcon build --symlink-install --cmake-args -DROS_EDITION=ROS2 -DDISTRO_ROS=jazzy -DCMAKE_BUILD_TYPE=Release
   source install/setup.bash
   ```

   The summary should read `3 packages finished`: `livox_ros_driver2`,
   `fast_lio` and `hexapod_bringup`. Compiler warnings from the two third-party
   packages are expected. So is a `hexapod_bringup` warning that `DISTRO_ROS`
   and `ROS_EDITION` were not used: those two settings are for the driver, and
   `colcon` passes them to every package.

These steps were last checked on 2026-11-16 in WSL 2 with Ubuntu 24.04.5,
GCC 13.3.0 and CMake 3.28.3.

## Replay a bag

`replay.launch.py` plays a bag's raw lidar and motion-sensor topics through
FAST-LIO, builds a Nav2 obstacle map (costmap) from FAST-LIO's point cloud,
and stops when the bag ends. It needs `ros-jazzy-nav2-costmap-2d` and
`ros-jazzy-nav2-lifecycle-manager`. The settings are in the launch file and
`config/costmap.yaml`.

```sh
ros2 launch hexapod_bringup replay.launch.py bag:=<input bag> \
  roll:=<rad> pitch:=<rad> height:=<m> rviz:=true
```

FAST-LIO's frame starts at the sensor's pose and is not levelled, so a tilted
sensor makes the floor look like an obstacle. `roll` and `pitch` level it and
`height` puts the map's `odom` frame on the floor. They describe how a
recording was made, not the robot. For the two 2026-11-14 bags, use
`roll:=-0.0088 pitch:=0.8007 height:=1.27`: the roll and pitch come from the
average accelerometer reading at rest, and the height from the floor's height
in the levelled point cloud.

To keep FAST-LIO's output or the map, start a recorder in another shell first,
for example `ros2 bag record -o <output bag> /Odometry /tf /costmap`.

The costmap process exits with a segmentation fault at shutdown, after the
bag has finished; the map it produced is unaffected.

Any other FAST-LIO running on the same ROS domain also processes the replayed
scans and doubles the output. Stop it first, or run the replay with a different
`ROS_DOMAIN_ID`.
