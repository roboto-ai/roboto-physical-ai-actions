# Contract reference

The `roboto-to-lerobot` action is driven by a **contract YAML** that lives in
the invocation dataset. It tells the action which ROS topics to read, how to
extract values from each message, how to align streams onto a common timeline,
and what transforms to apply along the way. Each event or session in the input
collection becomes one episode in the output LeRobot dataset; the contract is applied
identically to every episode.

## Table of contents

- [Minimal example](#minimal-example)
- [Top-level fields](#top-level-fields)
- [Observation specs](#observation-specs)
- [Video / image specs](#video--image-specs)
- [Action specs](#action-specs)
- [Task specs](#task-specs)
- [Selectors and `lerobot_names`](#selectors-and-lerobot_names)
- [Alignment](#alignment)
- [Transforms](#transforms)
- [Role-based binding](#role-based-binding)
- [Supported message types](#supported-message-types)
- [Validation rules and common errors](#validation-rules-and-common-errors)

## Minimal example

```yaml
name: my_task
version: 1
fps: 20
robot_type: my_arm

observations:
  - key: observation.images.exo
    topic: /camera/exo/image_raw/compressed
    type: sensor_msgs/msg/CompressedImage
    image:
      resize: [480, 640]   # [height, width]
    align: {method: nearest, tolerance_ms: 100}

  - key: observation.state
    topic: /robot/joint_states
    type: sensor_msgs/msg/JointState
    selector:
      names:    [joint_1, joint_2, joint_3]
      lerobot_names: [arm_1, arm_2, arm_3]
    align: {method: hold, tolerance_ms: 100}

actions:
  - key: action
    topic: /robot/joint_commands
    type: sensor_msgs/msg/JointState
    selector:
      names:    [joint_1, joint_2, joint_3]
      lerobot_names: [arm_1, arm_2, arm_3]
```

## Top-level fields

| Field               | Type    | Default       | Description                                                                                                                                                          |
|---------------------|---------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`              | string  | `"contract"`  | Logical contract name. Recorded with each episode for traceability.                                                                                                  |
| `version`           | int     | `1`           | Contract version. Bump when changing the schema in a way that breaks downstream consumers.                                                                           |
| `fps`               | float   | `20.0`        | Target sampling frequency, in Hz. The reference timeline (one frame per `1/fps` seconds) is built at this rate, and every observation/action is aligned onto it.     |
| `robot_type`        | string  | `null`        | Free-form label written into LeRobot dataset metadata.                                                                                                               |
| `action_lead_steps` | int     | `0`           | Shift action sampling N frames into the future to compensate for control delay. `0` means action[t] is read at the same instant as observation[t].                   |
| `observations`      | list    | `[]`          | Observation streams. Entries with an `image:` block are routed to the video pipeline; entries without become numeric observation features. See below.                |
| `actions`           | list    | `[]`          | Action streams. Become numeric action features.                                                                                                                      |
| `tasks`             | list    | `[]`          | Optional task channels (e.g., language prompts). The first declared spec's first in-window `std_msgs/msg/String` message becomes the episode's LeRobot `task` label; see [Task specs](#task-specs).|

## Observation specs

Each entry in `observations:` describes one stream. There are two flavors,
distinguished by whether an `image:` block is present:

- **Numeric observations** (no `image:` block) become a single LeRobot feature
  per `key`. Multiple entries with the same `key` are concatenated along the
  feature axis (see [Selectors](#selectors-and-lerobot_names) for naming).
- **Video / image observations** (with `image:` block) become a `dtype: video`
  feature in the LeRobot dataset. See [Video / image specs](#video--image-specs).

| Field           | Required | Description                                                                                                                                               |
|-----------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `key`           | yes      | LeRobot feature key. Conventionally `observation.state`, `observation.images.<name>`, etc.                                                                |
| `topic`         | one of   | ROS topic name. Mutually exclusive with `role`.                                                                                                           |
| `role`          | one of   | Role label resolved against file metadata at runtime. See [Role-based binding](#role-based-binding). Not allowed for image streams.                       |
| `type`          | yes      | Message type the topic carries. Picks the decoder, except where `image.type` does. See [Supported message types](#supported-message-types).               |
| `selector`      | no       | `{names: [...], lerobot_names: [...]}`. Selects which fields the decoder extracts. See [Selectors](#selectors-and-lerobot_names).                          |
| `image`         | no       | Image / video options: `resize: [H, W]`, and `field` and `type` for a nested image. See [Video / image specs](#video--image-specs).                       |
| `align`         | no       | `{method, tolerance_ms}`. Defaults to `hold` with the auto-bounded tolerance (`max(2/fps, 50 ms)`). See [Alignment](#alignment).                           |
| `transforms`    | no       | List of transforms applied to the stream. See [Transforms](#transforms).                                                                                   |

## Video / image specs

A spec under `observations:` becomes a video feature when it has an `image:`
block. The output dtype is always `video` (HWC uint8 RGB at the configured
resize). Each frame read from an image or compressed-video message, nested
ones included, is placed at the log time of that message: when it reached the
recorder.

```yaml
observations:
  - key: observation.images.exo
    topic: /camera/exo/image_raw/compressed
    type: sensor_msgs/msg/CompressedImage
    image:
      resize: [480, 640]   # [height, width]
    align: {method: nearest, tolerance_ms: 100}
```

`image:` block fields:

| Field    | Required | Description                                                                                                                                                   |
|----------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `resize` | yes      | `[height, width]` in pixels. Frames are resized to this exact shape.                                                                                          |
| `field`  | no       | Message path of an image nested in a custom message. See [Images nested in a custom message](#images-nested-in-a-custom-message).                             |
| `type`   | with `field` | Type of the image message at `field`. Required with `field`, rejected without it.                                                                         |

Restrictions on image streams:

- `align.method: linear` is rejected — use `hold` or `nearest`.
- `role:` binding is not allowed; image streams must specify `topic:` literally.

### Images nested in a custom message

Some recordings publish no topic per camera. Instead, one custom message
carries every camera, each as a whole image message in a field of its own.
Point a spec at one of those fields with `image.field`, the message path of
the nested image: field names joined by dots, as Roboto writes message paths.
As on every spec, `type:` names what the topic carries, here the custom
message. `image.type` names the image message at the path. Declare one spec
per camera:

```yaml
observations:
  - key: observation.images.front
    topic: /robot/observation
    type: my_robot_msgs/msg/Observation   # what the topic publishes
    image:
      field: cameras.front                    # message path of the nested image
      type: sensor_msgs/msg/CompressedImage   # what sits at that path
      resize: [480, 640]
  - key: observation.images.wrist
    topic: /robot/observation
    type: my_robot_msgs/msg/Observation
    image:
      field: arm.cameras.wrist
      type: sensor_msgs/msg/Image
      resize: [480, 640]
```

Each nested image is decoded exactly as the same image would be if it were
published on its own topic. The still images (`CompressedImage`, `Image`)
nested in one topic are read with a single fetch of that topic. Each nested
compressed-video stream is fetched and decoded on its own.

`image.type` may be `sensor_msgs/msg/CompressedImage`, `sensor_msgs/msg/Image`
or a [compressed-video type](#compressed-video), and may be written with or
without `/msg/` (`sensor_msgs/CompressedImage` names the same type).
File-backed video (`video`, `avi_video`, `mp4_video`) is read as whole frames
rather than by message path, so it cannot be nested.

Before it fetches any data, the converter checks each topic that a nested
image is read from. The conversion fails, with an error that names the spec
and the topic, if the topic breaks any of these rules:

- the topic's recorded type must be the spec's `type`;
- the type Roboto recorded at the path, if any, must be `image.type`;
- the topic must have the nested image's message paths. If it lacks them, the
  error lists the paths in that topic that have the fields of an `image.type`
  message.

`gen-node` supports nested images. The generated node subscribes to the topic
with the custom message from `type:`, once for all the cameras nested in it,
and decodes each camera from its path. The robot must have the package that
defines the custom message installed, so that the node can import it. As for
an image on its own topic, `gen-node` refuses a nested compressed-video
stream.

## Action specs

Action specs follow the same shape as numeric observations:

```yaml
actions:
  - key: action
    topic: /robot/joint_commands
    type: sensor_msgs/msg/JointState
    selector:
      names:    [joint_1, joint_2]
      lerobot_names: [arm_1, arm_2]
    align: {method: hold, tolerance_ms: 100}
```

Same fields as observations except no `image:` block. As with observations,
multiple entries with the same `key` are concatenated along the feature axis.

Set `topic:` and `type:` directly on the action spec, as above. The legacy
nested `publish: {topic, type}` form is no longer supported — see
[Validation rules and common errors](#validation-rules-and-common-errors).

## Task specs

Optional. `tasks:` streams supply the per-episode LeRobot `task` string —
the free-text label attached to every frame in an episode (e.g. a language
instruction).

```yaml
tasks:
  - key: prompt          # optional; defaults to topic
    topic: /task/prompt
    type: std_msgs/msg/String
```

Resolution runs once per episode, in this order:

1. **The Roboto event's own `task` metadata**, when present and non-empty.
   This always wins and the `tasks:` stream below is not even consulted.
2. **The `tasks:` stream**, when the event carries no `task` metadata: the
   payload of the *first* `std_msgs/msg/String` message from the *first*
   declared task spec whose timestamp falls within the episode's window. If
   more than one spec is declared, only the first is used and a warning
   names the ones ignored.
3. **`"default"`**, when neither of the above yields a value — no event
   metadata, no `tasks:` block, or no in-window message on the first spec's
   topic.

Only `std_msgs/msg/String` is meaningful here; a task spec pointed at a
different message type raises a `ValueError` naming the spec if its decoded
payload isn't a string.

## Selectors and `lerobot_names`

`selector:` controls which fields the decoder pulls out of each message:

```yaml
selector:
  names:          [joint_1, joint_2, joint_3]   # what to extract
  lerobot_names:  [arm_1,   arm_2,   arm_3]     # how to label them in LeRobot (optional)
```

Two rules to keep in mind:

1. **`names` semantics depend on the message type.** For `JointState` they're
   joint names (with optional `position.<joint>` / `velocity.<joint>` /
   `effort.<joint>` prefix; bare names default to `position`). For `Imu`,
   `Odometry`, `Twist` they're dotted paths into the message
   (e.g., `linear.x`). For `string_typed_msg` they're Series index keys.
   See [Supported message types](#supported-message-types) for the full list.

2. **`lerobot_names` is optional but strongly recommended.** Without it, the
   feature is labeled `<topic>/<selector_name>`, which is verbose and ties
   feature names to ROS topology. With it, you get clean LeRobot-facing names.
   With `names`, `lerobot_names` must have the same length as `names`, and
   every `lerobot_name` must be unique across the entire contract.

When multiple specs share a `key` (e.g., two `observation.state` entries from
two arms), all `lerobot_names` from those specs are concatenated in order to
form the names of the combined feature.

### Labels by position

Some messages carry no names to select by: a `JointState` whose `name` field
is empty, or holds a single label that names no joint, or a
`Float64MultiArray`. Without `names`, the decoder emits the whole vector (for
`JointState`, every `position`), and `lerobot_names` alone labels it by
position: the first name labels the first value, and so on.

```yaml
observations:
  - key: observation.state
    topic: /arm/joint_states        # JointState with an empty name field
    type: sensor_msgs/msg/JointState
    selector:
      lerobot_names: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
```

The names reach the dataset's `meta/info.json` as they would with `names`,
including concatenation across specs that share a `key`, and must still be
unique across the contract.

- Every message must decode to exactly as many values as there are
  `lerobot_names`. A message with more or fewer values fails the conversion,
  and an observation message fails the live node too. The error names the
  spec's key, its topic and both lengths, so no value is labelled wrongly.
- Only types whose decoder, without `names`, emits numbers in an order the
  message fixes accept labels by position: `JointState`, `JointTrajectory`,
  `MultiDOFCommand`, `Imu`, `Odometry`, `Twist`, the `*MultiArray` and scalar
  `std_msgs` types, `AnymalState` and `SeActuatorReadings`. The vector each
  emits is described in [Numeric streams](#numeric-streams). The contract
  loader refuses `lerobot_names` without `names` on any other type, such as
  `std_msgs/msg/String`, `string_typed_msg` or an image.
- To keep only some of the values, select them with `names`.

## Alignment

Each non-image stream is merged onto the reference timeline (one row per
`1/fps` seconds) using its `align:` block:

```yaml
align: {method: hold, tolerance_ms: 100}
```

| Method     | Behavior                                                                                                |
|------------|---------------------------------------------------------------------------------------------------------|
| `hold`     | Last-observation-carried-forward (backward as-of join). Good for slow state signals. **Default.**       |
| `nearest`  | Pick the closest sample in time, regardless of direction. Good for cameras and high-rate signals.       |
| `linear`   | Linearly interpolate between the two bracketing samples. Numeric streams only — not allowed for images. |
| `none`     | Exact timestamp matches only; everything else becomes NaN.                                              |

`tolerance_ms` is the maximum gap (in milliseconds) between the reference
frame timestamp and the matched sample before the result becomes NaN. Set
this to a value matching your signal's expected period — e.g., `100` for a
10 Hz signal.

| `tolerance_ms`             | Meaning                                                                                                          |
|----------------------------|-------------------------------------------------------------------------------------------------------------------|
| omitted, or `align:` omitted entirely | Auto-bounded default: `max(2/fps, 50ms)`. Matches the bound the live runtime independently applies to an unlimited tolerance, so offline and live agree without either side special-casing the other. |
| `null`                     | Unlimited — always carry forward / always pick nearest, however far.                                            |
| `0`                        | **Rejected.** `0` used to mean "unlimited" in older contracts; that spelling collided with "zero tolerance" and is no longer accepted. Use `null` for unbounded, or omit `tolerance_ms` for the auto-bounded default. |
| a positive number          | That bound, unchanged.                                                                                            |

The legacy field names `strategy` and `tol_ms` are no longer accepted — use
`method` and `tolerance_ms`.

## Transforms

Transforms are applied per-stream in the order they appear, before the stream
becomes part of the LeRobot dataset:

```yaml
actions:
  - key: action
    topic: /robot/joint_commands
    type: sensor_msgs/msg/JointState
    selector: {names: [j1, j2], lerobot_names: [arm_1, arm_2]}
    transforms:
      - type: resample_uniform
        stage: pre
      - type: butterworth_lowpass
        stage: pre
        cutoff_hz: 30
        order: 2
      - type: finite_difference     # stage defaults to "post"
```

### Stages

| Stage   | When it runs                              | Notes                                                                                  |
|---------|-------------------------------------------|----------------------------------------------------------------------------------------|
| `pre`   | Before alignment, on the raw stream.      | Operates on raw timestamps. Use for resampling and pre-alignment filtering.            |
| `post`  | After alignment, on the reference timeline. | Operates at `contract.fps`. **Default when `stage:` is omitted.**                  |

Any other `stage` fails when the contract loads.

### Columns

A transform applies to every column of its stream unless it lists `columns`,
by their names in `selector.names`; the other columns pass through unchanged.
`columns` must be a non-empty list of distinct names, each in
`selector.names`, or the contract fails when it loads. A transform given
`columns` must keep the number of rows and columns it is given, so
`resample_uniform` cannot take them.

Frames where a stream has a gap longer than its tolerance are dropped. A
`post` transform runs separately on each unbroken run of the remaining
frames, so `finite_difference` and the Butterworth filters never treat the
two sides of a gap as neighbours. A run too short for `butterworth_lowpass`'s usual
edge padding is filtered with as much padding as it allows, and a single frame's
`finite_difference` is zero.

### Built-in transforms

| Type                    | Stage     | Params                                            | Description                                                                                                              |
|-------------------------|-----------|---------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `resample_uniform`      | `pre`     | —                                                 | Resamples an irregularly sampled signal onto a uniform grid (median inter-sample interval) using nearest-neighbour.       |
| `butterworth_lowpass`   | any       | `cutoff_hz` (required), `order` (default `2`)     | Zero-phase Butterworth low-pass filter (uses scipy `sosfiltfilt`).                                                       |
| `butterworth_lowpass_causal` | any  | `cutoff_hz` (required), `order` (default `2`), `fs_hz` (optional) | Causal (forward-only) Butterworth low-pass filter (uses scipy `sosfilt`, seeded with `sosfilt_zi` to avoid a startup transient). |
| `finite_difference`     | any       | —                                                 | Frame-to-frame difference (`out[i] = in[i+1] - in[i]`). Last row is repeated to keep the shape constant.                  |
| `quaternion_continuity` | `pre`     | —                                                 | Takes 4 quaternion columns, in any order. Negates each quaternion whose dot product with the one before it is negative, so the stream never jumps from q to -q. |
| `quaternion_normalize`  | `post`    | —                                                 | Takes 4 quaternion columns. Scales each quaternion to unit length; raises if one has norm below 0.7, which means `quaternion_continuity` is missing. |
| `unwrap_angle`          | `pre`     | `degrees` (default `false`)                       | Removes the full-turn jumps where an angle wraps, so it changes continuously. Assumes less than half a turn between samples. |
| `wrap_angle`            | `post`    | `degrees` (default `false`)                       | Brings an angle back into [-π, π) radians or [-180, 180) degrees.                                                        |
| `polar_to_cartesian`    | `pre`     | `degrees` (default `false`)                       | Takes `[r, theta]`, theta from +x towards +y, and returns `[x, y]`.                                                       |
| `spherical_to_cartesian` | `pre`    | `degrees` (default `false`)                       | Takes `[r, theta, phi]`, theta the inclination from +z and phi the azimuth from +x towards +y, and returns `[x, y, z]`. |

`fs_hz` (on `butterworth_lowpass_causal` only) declares the sample rate the
filter is *designed* for, overriding the `fps` the transform would otherwise
be called with. This is what makes the causal filter reproducible in the
live runtime: for a `stage: pre` transform, offline training designs the
filter with the *measured* raw topic rate (see `raw_fs` in `lerobot.py`), a
number the live runtime — which never sees the offline episode — cannot
recompute on its own. Declaring `fs_hz` freezes that rate into the contract
so both sides build identical `sos` coefficients; the live adapter refuses
to boot a `stage: pre` causal transform with no declared `fs_hz`. When
`fs_hz` deviates from the stream's actual measured rate by more than 10%, a
`WARNING` names both numbers — a wrongly declared rate is a silent parity
break, but a hard error would be too aggressive for streams with
legitimately irregular timing.

`butterworth_lowpass` vs `butterworth_lowpass_causal`: the zero-phase variant
(`sosfiltfilt`) has no phase distortion but is *non-causal* — each output
sample depends on both past and future samples in the episode, so it cannot
be reproduced online where a deployed policy only ever has past samples.
The causal variant (`sosfilt`) is forward-only, so the identical filter can
run sample-by-sample at inference time and reproduce exactly what training
saw — at the cost of phase lag / group delay. Prefer `butterworth_lowpass`
when only offline smoothing matters (e.g. cleaning an observation stream
that plays no role in the deployed policy's own filtering); prefer
`butterworth_lowpass_causal` for any stream where training and deployment
must apply the identical filter. Both raise a `ValueError` naming the
offending `fs` if `cutoff_hz` is not strictly between `0` and `fs/2`
(Nyquist) — remember `fs` is the *measured raw rate* for `stage: pre`
transforms, not `contract.fps` (or the declared `fs_hz`, when set).

**Live runtime.** Only `butterworth_lowpass_causal` can run in the live
runtime (`roboto_to_lerobot.runtime.live_adapter.LiveAdapter`) —
`butterworth_lowpass` needs future
samples and can never run online, so `gen-node` and the live adapter both
refuse it (and `resample_uniform`, and every other transform) unconditionally.
Both also refuse `butterworth_lowpass_causal` with `columns` on an
observation, since the live filter applies to every column of a stream.
On observations, the causal filter runs for real: a `stage: pre` instance
filters each decoded message before it enters the alignment buffer (so it
must declare `fs_hz`, per above), and a `stage: post` instance filters at
policy-tick rate, designed with `contract.fps`. On actions, a declared
`butterworth_lowpass_causal` is accepted but applied as a **no-op
pass-through** on the policy's published output: the converter filtered the
*recorded* action stream to build the training targets, so the policy was
trained to emit values already in the filtered target space, and a low-pass
filter has no stable inverse to undo a filter that was never applied to the
policy's own output. On video/image streams, any transform — causal or not
— is refused; the runtime treats video frames as opaque decoded arrays.

When chaining a `pre`-stage timestamp-modifying transform (`resample_uniform`)
with later filters, the `fps` argument passed to subsequent transforms is
recomputed from the new timestamps automatically — so a `butterworth_lowpass`
after `resample_uniform` sees the resampled rate, not the raw one.

### Rotations, angles and polar vectors

`align: linear` blends each column on its own, which is wrong for these as
recorded: a quaternion and its negation are the same rotation, an angle
jumps by a full turn where it wraps, and a polar vector's angle wraps too.
Pair a `pre` transform that makes the stream safe to blend with, where it
has one, a `post` transform that restores the recorded form:

```yaml
observations:
  - key: observation.state
    topic: /tool/odom
    type: nav_msgs/msg/Odometry
    selector:
      names: [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
              pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
      lerobot_names: [x, y, z, qx, qy, qz, qw]
    align: {method: linear, tolerance_ms: 100}
    transforms:
      - type: quaternion_continuity
        stage: pre
        columns: [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
      - type: quaternion_normalize      # stage defaults to "post"
        columns: [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
```

- **Quaternions.** `quaternion_continuity` before alignment and
  `quaternion_normalize` after it give normalized linear interpolation, which
  follows the same path as SLERP. Its angle from SLERP is about
  1.2e-6 × θ³ degrees, θ being the rotation in degrees between the two
  samples blended: under 0.001° at 10°, 0.06° at 36°, 0.5° at 72°.
- **Single angles** (yaw, a joint that turns without limit): `unwrap_angle`
  before alignment, `wrap_angle` after it. Interpolating Euler angle triples
  is not recommended; record a quaternion instead.
- **Polar and spherical vectors:** `polar_to_cartesian` or
  `spherical_to_cartesian` before alignment. The dataset then holds the
  cartesian vector, which has no wrap or pole for a policy to learn around;
  set `lerobot_names` to name its components.

None of these can run in the live runtime, which refuses them like every
transform other than `butterworth_lowpass_causal`.

To add a new transform, register it in
`src/roboto_to_lerobot/transforms.py` via `@register_transform("name")`.

## Role-based binding

In multi-device datasets where the same logical role (e.g., "left arm",
"exo camera") may live under different topic names from one recording to the
next, you can bind a spec by **role** instead of topic. The role is resolved
at runtime against the per-file `role` metadata tag. Any action or script
may write it.

```yaml
observations:
  - key: observation.state
    role: left_arm                  # instead of `topic: ...`
    type: sensor_msgs/msg/JointState
    selector: {names: [j1, j2, j3], lerobot_names: [l1, l2, l3]}

actions:
  - key: action
    role: left_arm
    type: sensor_msgs/msg/JointState
    selector: {names: [j1, j2, j3], lerobot_names: [l1, l2, l3]}
```

How resolution works:

1. The action lists every file in the dataset and buckets them by their
   `metadata["role"]` tag.
2. For each role-bound spec, exactly one file must carry the matching role
   tag. Zero matches or multiple matches both raise an error.
3. On that file, exactly one topic must cover all `selector.names` in its
   message paths. Zero or multiple candidate topics also raise an error.

Limits:

- Image / video specs cannot be role-bound (image topics are pinned to a
  single file URL, which the role layer doesn't model).
- A spec must set **exactly one of** `topic` or `role` — never both, never
  neither.

To use role binding end-to-end:

1. Tag each source file with a `role: <label>` metadata field.
2. Author the contract with `role:` in place of `topic:`.
3. Invoke `roboto-to-lerobot` as usual.

## Supported message types

`type:` dispatches to a registered decoder (`image.type` does, for an
[image nested in a custom message](#images-nested-in-a-custom-message)). The
selector semantics depend on the type — see below. To add a new decoder,
register it in `src/roboto_to_lerobot/decoders.py` via
`@register_decoder("type")`.

### Image / video

| `type:`                          | Selector semantics       | Required `image:` fields                              |
|----------------------------------|--------------------------|-------------------------------------------------------|
| `sensor_msgs/msg/CompressedImage`| not used                 | `resize`                                              |
| `sensor_msgs/msg/Image`          | not used                 | `resize`                                              |
| `video`, `avi_video`, `mp4_video`| not used                 | `resize`. File-backed video frames (one frame per row).|
| `foxglove_msgs/msg/CompressedVideo`| not used               | `resize`. Compressed video (H.264/H.265/VP9/AV1).      |

Supported `Image` encodings: `rgb8`, `bgr8`, `mono8`, `rgba8`, `bgra8`, `8UC1`
(decodes as mono8).

While an episode is built and written, its camera messages wait in a file on
the action's local disk rather than in memory, and each frame is read back
from that file just before it is decoded and written. The file is deleted as
soon as its episode is written or skipped. These files take as much disk as
the recorded camera messages of the episodes that are being built or waiting
to be written: up to `pool_size` + 4 episodes in each shard.

#### Compressed video

Topics that Roboto ingestion tags `compressedVideo` store one encoded video
*access unit* per message rather than a standalone still image, so a frame in
the middle of a group of pictures (GOP) can only be decoded together with the
frames back to its keyframe. The converter handles that: it decodes each
episode's whole time range in one pass, walking back up to 10 s to find the
keyframe that anchors the range. The pass runs twice: once while fetching,
to learn which frames decode, and again while the episode is written, keeping
only the few decoded frames about to be written. A camera's memory use
therefore does not grow with the episode's length. Declare the topic's real
schema name (`foxglove_msgs/msg/CompressedVideo`, `foxglove_msgs/CompressedVideo`, or
`foxglove.CompressedVideo`) — declaring such a topic as `CompressedImage` is
refused with an explanatory error rather than silently producing broken frames.

Notes and limits:

- Decoding needs the `roboto[video]` extra (PyAV); it ships with the action image.
- The codec is read from each message's `format` field. A codec outside
  H.264/H.265/VP9/AV1 fails the conversion instead of dropping the camera.
- Frames whose keyframe is unreachable are skipped with a warning; a range where
  *nothing* decodes fails the conversion.
- `gen-node` refuses compressed video: a live ROS node holds no GOP state and
  cannot decode a stream message by message. Use an image topic for live
  inference.

### Numeric streams

| `type:`                                   | What `selector.names` means                                                                                                |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `sensor_msgs/msg/JointState`              | Joint name. Optional `position.<joint>` / `velocity.<joint>` / `effort.<joint>` prefix; bare name defaults to `position`. |
| `trajectory_msgs/msg/JointTrajectory`     | Joint name. Each trajectory point becomes its own row at `header.stamp + time_from_start`. **Action streams only.**       |
| `control_msgs/msg/MultiDOFCommand`        | DOF name. Optional `values.<dof>` / `values_dot.<dof>` prefix; bare name defaults to `values`.                            |
| `sensor_msgs/msg/Imu`                     | Dotted path into the message, e.g., `orientation.x`, `angular_velocity.z`. Without `names`: returns `[quat, ang_vel, lin_acc]` (10 values). |
| `nav_msgs/msg/Odometry`                   | Dotted path. Without `names`: returns `[pos.xyz, quat.xyzw]` (7 values).                                                  |
| `geometry_msgs/msg/Twist`                 | Dotted path, e.g., `linear.x`. Without `names`: returns `[linear.xyz, angular.xyz]` (6 values).                           |
| `std_msgs/msg/Float32MultiArray`          | not used; the full `data` array is emitted as `float32`.                                                                  |
| `std_msgs/msg/Float64MultiArray`          | not used; full `data` as `float64`.                                                                                       |
| `std_msgs/msg/Int32MultiArray`            | not used; full `data` as `int32`.                                                                                         |
| `std_msgs/msg/Float32` / `Float64` / `Int32` / `Int64` | not used; single-element array containing `data`.                                                              |
| `std_msgs/msg/String`                     | not used; emitted as a Python string (intended for `tasks:`).                                                              |
| `string_typed_msg`                        | Series index keys (treats each index entry as a float-castable field). `selector.names` is required.                      |
| `anymal_msgs/AnymalState`                 | ANYmal joint name into `joints`. Optional `position.<joint>` / `velocity.<joint>` / `acceleration.<joint>` / `effort.<joint>` prefix; bare name defaults to `position`. Without `names`: all joint positions in message order. |
| `series_elastic_actuator_msgs/SeActuatorReadings` | ANYmal joint name, mapped **positionally** (the wire message has no per-actuator name) to the fixed order `LF/RF/LH/RH × HAA/HFE/KFE`. Optional field path prefix, e.g. `commanded.velocity.<joint>` / `state.joint_position.<joint>`; bare name defaults to `commanded.position` (the policy setpoint). Without `names`: all 12 `commanded.position`. |

## Validation rules and common errors

The contract loader checks a few things up front. If you hit one of these,
the error message will name the offending spec.

- **Exactly one of `topic` / `role`** must be set on each observation/action.
- **Image streams cannot use `role:`** or `align.method: linear`.
- **`lerobot_names` must match `names` in length**, and every `lerobot_name`
  in the contract must be globally unique. Both errors name the offending
  spec(s) — a duplicate name names both conflicting specs (key + topic/role),
  and a length mismatch prints both lists. Without `names`, `lerobot_names`
  label values by position, which only some types accept — see
  [Labels by position](#labels-by-position).
- **`align.method`** must be one of `hold`, `nearest`, `linear`, `none`.
- **`align.tolerance_ms: 0`** is rejected — see [Alignment](#alignment).
- **Legacy `strategy`/`tol_ms` align keys and the nested `publish:` action
  form** are rejected — see [Alignment](#alignment) and
  [Action specs](#action-specs).
- **`image.field` and `image.type`** must be set together. `image.field` must
  be a message path (field names joined by dots), and `image.type` must name
  an image or compressed-video message, not file-backed video (`video`,
  `avi_video`, `mp4_video`) — see
  [Images nested in a custom message](#images-nested-in-a-custom-message).
- **An `image.depth:` block is rejected** — depth images are not supported
  yet; remove the block. If you need depth support, open an issue or a PR at
  [the repository](https://github.com/roboto-ai/roboto-physical-ai-actions).
- A missing or wrong-typed required field on a spec (e.g. no `key` or no
  `type`) raises a `ValueError` naming that spec's position and key, e.g.
  `observations[2] (key 'observation.state'): ...`, instead of a bare
  `KeyError`/`TypeError`.
- At runtime: every topic referenced by the contract must be present on at
  least one file in the dataset, otherwise the action errors out before
  doing any work.

Tips:

- `align.tolerance_ms` defaults to `max(2/fps, 50ms)` when omitted — that's
  usually a reasonable starting point. Tighten it toward ~1 period of your
  slowest signal (e.g., `100` for a 10 Hz signal) if you need to catch
  gaps explicitly; too tight and you'll see NaN gaps, too loose (or `null`
  for unlimited) and you'll silently carry stale samples into long pauses.
- Prefer `hold` for slow state signals and `nearest` for cameras. Use
  `linear` only for smooth numeric signals where interpolation is meaningful.
- If you're chaining transforms that change the time axis, place
  `resample_uniform` first (`stage: pre`) and any low-pass filter after it.
