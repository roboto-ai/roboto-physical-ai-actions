#!/usr/bin/env bash

# Run inside the ros:humble-ros-base container built from
# Dockerfile.ros2_humble. Drives the pipeline:
#   gen-node → colcon build → (ros2 bag play + ros2 run +
#   ros2 topic echo on the action and obs topics) → verify_action.py
#
# Inputs (all bind-mounted by run.sh):
#   /smoke/contract.yaml      — codegen-compatible contract
#   /smoke/ros_pkg/           — ament_python package skeleton; gen-node
#                               output goes into ros_pkg/inference_node/node.py
#   /fixture.mcap             — the MCAP fixture (fetched on the host)
#
# Outputs:
#   exit 0 — the recorded action matched the stub's obs[:8] + offset
#            invariant within the verifier's tolerance
#   exit 1 — gen-node, colcon build, the action echo, or the verifier
#            failed/timed out
#
# Logs go to /tmp/{bag,node,echo,obs}.log; on failure they're tail'd to
# stderr so the host wrapper sees what happened without needing to
# spelunk the container.

set -eo pipefail

# `set -u` (nounset) is deliberately omitted: ROS 2's setup.bash and
# colcon's generated setup.bash both read variables they have not
# defined yet (AMENT_TRACE_SETUP_FILES, _colcon_prefix_*, etc.) and
# explode under `set -u`. Sourcing them is the first thing this script
# does and there is no read-only "wrapped source" idiom that survives
# every ROS 2 distro. Local variable typos lose nounset's safety net
# in exchange — the script is short, so the trade is acceptable.

# Source the ROS environment so ``ros2``, ``colcon``, and rclpy are on PATH.
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

ACTION_TOPIC="/teleop/action"
OBS_TOPIC="/robot/joint_states"
FIXTURE_PATH="/fixture.mcap"
TIMEOUT_SECONDS="${ROS_SMOKE_TIMEOUT:-30}"
CONTRACT_PATH="/smoke/contract.yaml"
WS_ROOT="/smoke/ws"
GEN_NODE_OUT="/smoke/ros_pkg/inference_node/node.py"
MANIFEST_PATH="/smoke/ros_pkg/manifest.json"

if [[ ! -f "$FIXTURE_PATH" ]]; then
    echo "run_in_container: $FIXTURE_PATH missing — run fetch_fixture.py on the host first." >&2
    exit 1
fi

# Generate the manifest from the contract sha so verify_manifest passes.
# Mirrors what the converter writes at main.py:1422; the docker smoke
# does not run the converter, so we synthesize the minimum the
# generated node's verify_manifest call expects.
CONTRACT_SHA=$(sha256sum "$CONTRACT_PATH" | awk '{print $1}')
printf '{"contract": {"sha256": "%s"}}\n' "$CONTRACT_SHA" > "$MANIFEST_PATH"
echo "run_in_container: contract sha=$CONTRACT_SHA  manifest=$MANIFEST_PATH"

# Render the generated node into the colcon package. ``--force`` because
# previous iterations may have left a stale node.py from an earlier
# container run with the same bind-mounted ros_pkg/ directory.
#
# Use ``python3 -m roboto_to_lerobot.codegen.cli`` rather than the
# packaged ``roboto-to-lerobot`` console script — sourcing ros2's
# setup.bash on humble can shadow or unset PATH entries placed by
# pip's user-site install, and the explicit module path bypasses
# that fragility.
echo "run_in_container: rendering node into $GEN_NODE_OUT"
python3 -m roboto_to_lerobot.codegen.cli gen-node "$CONTRACT_PATH" \
    --policy-module inference_node.policy \
    --manifest "$MANIFEST_PATH" \
    --out "$GEN_NODE_OUT" \
    --force

# Build a colcon workspace that contains just the smoke package. ament
# build artefacts land under ws/install; sourcing ws/install/setup.bash
# adds the package to AMENT_PREFIX_PATH so ``ros2 run`` finds it.
mkdir -p "$WS_ROOT/src"
# symlink rather than copy so the bind-mounted ros_pkg/ is the source
# of truth for the entry-point shim (gives faster iteration if a dev
# adds a debug print).
if [[ ! -e "$WS_ROOT/src/inference_node" ]]; then
    ln -s /smoke/ros_pkg "$WS_ROOT/src/inference_node"
fi

# `--symlink-install` would have colcon ask setuptools for an editable
# install (`setup.py develop`). The smoke rebuilds the workspace from
# scratch on every container run, so the symlink/editable optimisation
# (faster re-iteration on source edits) buys nothing here — skip it.
# (The image deliberately pins apt's setuptools at 59.6.0; see
# Dockerfile.ros2_humble. We upgrade pip only, not setuptools, so the
# editable path would in fact still work — there is just no iteration
# to speed up.)
echo "run_in_container: colcon build"
(cd "$WS_ROOT" && colcon build --packages-select inference_node) >&2

# shellcheck disable=SC1091
source "$WS_ROOT/install/setup.bash"

# Add the rosbag2 MCAP storage plugin to the storage search path. The
# apt package drops the .so where rosbag2 already looks, but the
# environment variable below forces ros2 bag play to prefer it (the
# default is sqlite3 even on a .mcap file unless told otherwise).
export ROSBAG2_STORAGE_DEFAULT_FORMAT=mcap

# Start the generated node in the background; PYTHONUNBUFFERED so its
# stdout/stderr flush eagerly into /tmp/node.log (else a SIGTERM may
# truncate the tail we print on failure).
#
# The node keys both buffered arrivals and the timer tick on the SAME
# clock — `self.get_clock().now().nanoseconds` (template rclpy.py.j2:93
# and :98) — and never on `msg.header.stamp`. So freshness is measured
# on one timeline regardless of clock source: under plain wall-clock the
# arrivals and ticks stay mutually consistent (samples are milliseconds
# old, not stale), and sample() returns a value.
#
# use_sim_time:=true (fed by the bag's /clock via --clock below) is
# therefore not strictly required for the smoke to pass; it pins the
# node's clock to the bag's recorded time so the live tolerance_ms
# window behaves the way it will against a real driver publishing
# recorded/sim time, matching the offline converter's log-time alignment.
echo "run_in_container: starting generated node (use_sim_time=true)"
PYTHONUNBUFFERED=1 ros2 run inference_node node \
    --ros-args -p use_sim_time:=true \
    > /tmp/node.log 2>&1 &
NODE_PID=$!

# Subscribe to the action topic BEFORE bag play starts. ros2 topic echo
# --once blocks until it sees one message, then exits 0; piped through
# timeout(1) it bounds the wait deterministically.
echo "run_in_container: subscribing to $ACTION_TOPIC (timeout=${TIMEOUT_SECONDS}s)"
(timeout "$TIMEOUT_SECONDS" ros2 topic echo --once --qos-reliability reliable \
    "$ACTION_TOPIC" sensor_msgs/msg/JointState > /tmp/echo.log 2>&1) &
ECHO_PID=$!

# Parallel CONTINUOUS observation capture for verify_action.py. The
# verifier needs to find the obs the policy actually sampled — not
# just any obs — because end-effector joints move fast enough
# (~1.5 rad/sec) that even a few-hundred-ms gap between the captured
# obs and the policy's sample produces per-joint deltas an order of
# magnitude beyond any sane single-obs tolerance. Capturing the whole
# stream lets the verifier search for the matching obs (delta
# ~float-precision when the head of the stream is captured intact)
# instead of guessing within a loose tolerance. We subscribe before
# bag play starts (same as the action echo) so the bag-played stream
# from frame 0 lands in /tmp/obs.log; the kill below stops the
# capture after the action lands. QoS matches the generated node's
# subscription (JointState → qos_reliable per render.py).
echo "run_in_container: subscribing to $OBS_TOPIC (continuous, killed after action lands)"
(ros2 topic echo --qos-reliability reliable \
    "$OBS_TOPIC" sensor_msgs/msg/JointState > /tmp/obs.log 2>&1) &
OBS_ECHO_PID=$!

# Small sleep so the subscriber is up before bag play starts publishing.
# Without it, a fast fixture (few seconds of content) can complete
# before ros2 topic echo subscribes, which leaves echo waiting forever.
sleep 2

# Remap the bag's /teleop/action onto a side topic so the recorded
# teleop does not collide with the generated node's publisher. Without
# this, ros2 topic echo --once happily latches whichever message
# arrived first (usually the bag's recorded action), and the smoke
# would pass even with a broken generated node. Remapping forces the
# echo result to come from our publisher.
# --clock 100: publishes /clock at 100 Hz from the bag's recorded time
#          so the node (started with use_sim_time:=true) runs on the
#          bag's timeline — see the use_sim_time note above.
# --rate 2: replays at 2× recorded speed to shorten time-to-first-tick.
#          `ros2 topic echo --once` latches on the first matching
#          message and exits, so only time-to-first-publish gates the
#          smoke — not how much of the episode is replayed.
echo "run_in_container: ros2 bag play $FIXTURE_PATH (--clock, remapping bag's $ACTION_TOPIC away)"
ros2 bag play --storage mcap "$FIXTURE_PATH" \
    --clock 100 \
    --rate 2 \
    --remap "${ACTION_TOPIC}:=${ACTION_TOPIC}_bag_recorded" \
    > /tmp/bag.log 2>&1 &
BAG_PID=$!

# Wait for echo: exit 0 means a message landed, 124 means timeout,
# anything else is a real failure.
set +e
wait "$ECHO_PID"
ECHO_RC=$?
set -e

# Stop the continuous obs capture now that the action has landed (or
# we've timed out). SIGTERM is enough for ros2 topic echo to flush its
# YAML buffer; the `wait` swallows the non-zero "killed" exit code.
kill "$OBS_ECHO_PID" 2>/dev/null || true
wait "$OBS_ECHO_PID" 2>/dev/null || true

# Clean up background processes; suppress errors if they already exited.
kill "$BAG_PID" 2>/dev/null || true
kill "$NODE_PID" 2>/dev/null || true
wait "$BAG_PID" 2>/dev/null || true
wait "$NODE_PID" 2>/dev/null || true

if [[ "$ECHO_RC" -eq 0 ]]; then
    # Verifier: the stub policy in ros_pkg/inference_node/policy.py
    # computes action[i] = obs[i] + 100.0 (selector-ordered) for
    # i in 0..7. We captured the $OBS_TOPIC stream in parallel and
    # verify_action.py searches it for the obs the policy actually
    # sampled — reordering each bag JointState by the contract's
    # selector names so a permuted bag doesn't trip the gate by
    # coincidence. A pass proves observations actually flowed
    # through the runtime kernel into the policy.
    #
    # The action width 8 is pinned in FOUR coupled places: here,
    # contract.yaml's /teleop/action selector, policy.py's
    # _ACTION_WIDTH, and verify_action.py's _ACTION_WIDTH. The +100
    # offset is mirrored in policy.py (_STUB_OFFSET) and
    # verify_action.py (_STUB_OFFSET, plus _TOLERANCE). All change
    # together if the contract widens or the transform is altered.
    if ! grep -E "^position:" /tmp/echo.log >/dev/null 2>&1; then
        echo "run_in_container: FAILURE — action message had no 'position:' field" >&2
        head -80 /tmp/echo.log >&2
        exit 1
    fi
    if ! grep -E "^position:" /tmp/obs.log >/dev/null 2>&1; then
        echo "run_in_container: FAILURE — no observation captured on $OBS_TOPIC" >&2
        head -80 /tmp/obs.log >&2
        exit 1
    fi
    if ! python3 /smoke/verify_action.py \
            --obs-log /tmp/obs.log \
            --action-log /tmp/echo.log \
            --contract "$CONTRACT_PATH"; then
        echo "--- /tmp/echo.log (truncated) ---" >&2
        head -40 /tmp/echo.log >&2
        echo "--- /tmp/obs.log (truncated) ---" >&2
        head -40 /tmp/obs.log >&2
        exit 1
    fi
    echo "run_in_container: SUCCESS — input-dependent stub invariant holds on $ACTION_TOPIC"
    echo "--- /tmp/echo.log (truncated) ---"
    head -30 /tmp/echo.log || true
    exit 0
fi

echo "run_in_container: FAILURE (rc=$ECHO_RC)" >&2
echo "--- /tmp/node.log (tail) ---" >&2
tail -80 /tmp/node.log >&2 || true
echo "--- /tmp/bag.log (tail) ---" >&2
tail -40 /tmp/bag.log >&2 || true
echo "--- /tmp/echo.log (tail) ---" >&2
tail -40 /tmp/echo.log >&2 || true
echo "--- /tmp/obs.log (tail) ---" >&2
tail -40 /tmp/obs.log >&2 || true
exit 1
