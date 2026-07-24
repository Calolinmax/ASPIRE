
BEHAVIOR-1K

### Grasping Patterns

Patterns for grasping objects with the R1 Pro robot. Apply when writing or debugging grasp sequences in BEHAVIOR-1K tasks.

#### Pattern 1: `move_hand` vs `grasp_object`

##### Problem

Two grasp execution methods exist with very different tradeoffs:

* `move_hand(pose, arm)` — fast, predictable, but limited reach near floor
* `grasp_object(pregrasp, grasp, name, arm)` — slower, uses cuRobo IK+motion planner, better reach but can hang

##### When to Apply

* Always when choosing how to execute a grasp after `sample_grasp_pose`.

##### Decision Matrix

| Scenario                           | Use First          | Fallback         |
| ---------------------------------- | ------------------ | ---------------- |
| Floor object (z < 0.1m) at ≤0.45m | `grasp_object`   | `move_hand`    |
| Floor object at 0.5-0.6m           | `move_hand`      | `grasp_object` |
| Table/shelf object                 | `move_hand`      | `grasp_object` |
| Time budget < 200s                 | `move_hand` only | —               |
| Time budget > 400s                 | Both in sequence   | —               |

##### Timing Data (Soda Task)

| Method                         | Typical Duration | Worst Case             | Hang Risk |
| ------------------------------ | ---------------- | ---------------------- | --------- |
| `move_hand` (pregrasp+grasp) | 80-130s          | 200s                   | None      |
| `grasp_object`               | 150-265s         | **619s+ (hang)** | Yes       |

##### Code Template: Interleaved Per-Pose

```
for pose_idx, (pg, g) in enumerate(zip(pregrasp_poses, grasp_poses)):
    if success or time_left() < 60:
        break

    # Try move_hand first (fast)
    for arm in [0, 1]:
        if success or time_left() < 60:
            break
        try:
            open_gripper(arm=arm)
            move_hand(pg, arm=arm)
            time.sleep(0.5)
            move_hand(g, arm=arm)
            time.sleep(0.5)
            close_gripper(arm=arm)
            time.sleep(0.5)
            lift_arm(arm=arm)
            time.sleep(1)
            if check_object_in_hand(arm=arm):
                success = True
                break
            else:
                open_gripper(arm=arm)
                lift_arm(arm=arm)
        except Exception:
            try:
                open_gripper(arm=arm)
                lift_arm(arm=arm)
            except Exception:
                pass

    # Immediately try grasp_object if move_hand failed (don't defer!)
    if not success and time_left() > 120:
        for arm in [0, 1]:
            if success or time_left() < 120:
                break
            try:
                open_gripper(arm=arm)
                grasp_object(pg, g, target_object, arm=arm)
                lift_arm(arm=arm)
                if check_object_in_hand(arm=arm):
                    success = True
                    break
                else:
                    open_gripper(arm=arm)
            except Exception:
                try:
                    open_gripper(arm=arm)
                except Exception:
                    pass
```

##### Critical: Interleave, Don't Defer

In v1 (soda), we tried ALL `move_hand` attempts first, then `grasp_object` as final fallback. Result: `move_hand` burned 789s, only 61s left — `grasp_object` skipped entirely.

In v2, we interleave: per-pose, try `move_hand` (both arms), then immediately `grasp_object` (both arms). This ensures each method gets a chance while time permits.

##### Evidence

* Seed 34 (baseline success): `move_hand` at 0.5m distance — took 100s total, SUCCESS
* Seed 28 (baseline success): `grasp_object` at 0.65m — took 220s, SUCCESS
* Seed 29 (baseline fail): `grasp_object` —  **hung 619s** , TIMEOUT
* Seed 29 (v3 fix): `grasp_object` at 0.45m — took 158s, SUCCESS

---

#### Pattern 2: Dual-Arm Grasp

##### Problem

LLM-generated code often uses a single arm (`arm=0` or `arm=1`). The chosen arm may not reach the object from the current approach angle.

##### When to Apply

* Always. Try both arms for every grasp pose.

##### Strategy

```
for arm in [0, 1]:
    try:
        open_gripper(arm=arm)
        grasp_object(pregrasp, grasp, object_name, arm=arm)
        lift_arm(arm=arm)
        if check_object_in_hand(arm=arm):
            success = True
            break
    except Exception:
        open_gripper(arm=arm)
```

##### Evidence

* Seed 26 baseline: `arm=1` only → FAIL
* Seed 33 baseline: `arm=1` only → FAIL (fixed with `arm=0`)
* Seed 35 baseline: `arm=0` only → FAIL
* Baseline successes: seed 32 used `arm=1`, seed 28 used `arm=0` — neither arm is universally better.

##### Arm Selection Heuristic

* `arm=0` (left) tends to reach objects slightly to the robot's left
* `arm=1` (right) tends to reach objects slightly to the robot's right
* When in doubt, try `arm=0` first (slightly more success in our data)

---

#### Pattern 3: Floor Object Grasping

##### Problem

Floor-level objects (z < 0.1m) are the hardest to grasp. The arm must extend horizontally AND reach down, which limits the reachable workspace.

##### When to Apply

* Object z-position < 0.1m (from `get_object_pose`)

##### Strategy

1. Navigate to **0.45m** (not 0.5m or 0.6m)
2. Use **`grasp_object`** (cuRobo IK) as primary method — it can solve harder IK problems than `move_hand`
3. Try from **perpendicular approach angle** (±90°) — may provide better arm geometry

##### Why `move_hand` Fails on Floor Objects

`move_hand` uses a simpler motion planner that can't always solve the IK for floor-level poses:

* At 0.6m: `move_hand(grasp_z=0.094)` → `success=false` (can't reach)
* At 0.5m: `move_hand(grasp_z=0.094)` → `success=false` still
* At 0.5m: `move_hand(grasp_z=0.160)` → `success=true` but `z=0.16` is 9cm ABOVE the can → misses

`grasp_object` uses cuRobo which has a more capable IK solver and motion planner.

##### Evidence

* Seed 29: `move_hand` failed at every distance. `grasp_object` at 0.45m from perpendicular angle → SUCCESS.
* Seed 34 (baseline): `move_hand` worked at ~0.5m — but this seed had a more favorable approach geometry.

---

#### Pattern 4: Grasp Hang Mitigation

##### Problem

`grasp_object` can hang for 600+ seconds when the cuRobo motion planner gets stuck. A single hung call burns the entire time budget.

##### When to Apply

* Always wrap `grasp_object` in try/except
* Always check time budget before calling

##### Strategy

```
if time_left() > 120:  # never call with < 120s remaining
    try:
        grasp_object(pregrasp, grasp, object_name, arm=arm)
    except Exception as e:
        print(f"grasp_object error: {e}")
        open_gripper(arm=arm)  # clean up
```

##### Evidence

* Seed 29 baseline: `grasp_object` hung for **619s** → TimeoutError
* Seed 29 v2 fix: `grasp_object` completed in 158s and 264s (no hang) — hang is not deterministic
* Seed 28 baseline: `grasp_object` took 220s → SUCCESS

##### Limitations

* Cannot interrupt `grasp_object` once it starts (C++ internals ignore SIGALRM)
* The 120s minimum budget is a safety margin, not a guarantee — actual duration can exceed 300s

---

#### Pattern 5: `check_object_in_hand` False Positives

##### Problem

`check_object_in_hand` can return true when the object is momentarily grasped but drops during or after `lift_arm`. The reward function checks if the object is STILL in hand AND lifted above initial height.

##### When to Apply

* After any "successful" grasp, be aware the reward may still be 0.

##### Evidence (Soda Validation)

* Seed 52: `move_hand` → `check_object_in_hand` true → `reward=0` (dropped)
* Seed 61: Same pattern — grasped but dropped

##### Mitigation

No code-level fix available. The gripper physics and object shape determine whether the grasp is stable. Retry with a different pose pair if the first grasp reports success but the task isn't completed.
