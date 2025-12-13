/**
 * Three.js Hand Skeleton Module
 *
 * Provides 3D skeletal hand visualization driven by IMU sensor orientation.
 *
 * =====================================================================
 * COORDINATE SYSTEM MAPPING (Sensor → Hand Model)
 * =====================================================================
 *
 * SENSOR FRAME (Puck.js Accel/Gyro - LSM6DS3):
 *   +X → toward WRIST
 *   +Y → toward FINGERS
 *   +Z → INTO PALM
 *
 * HAND MODEL FRAME (Three.js):
 *   +X → toward PINKY (thumb on -X, right hand)
 *   +Y → FINGER EXTENSION direction
 *   +Z → PALM NORMAL (toward viewer when palm faces camera)
 *
 * MAPPING STRATEGY:
 * When sensor is FLAT (face up, palm up):
 *   - Sensor Z points UP (ceiling) → Hand +Z should point UP
 *   - Need 90° pitch offset to rotate palm from facing viewer to facing up
 *   - Need 180° yaw offset to point fingers away from viewer
 *   - Need 180° roll offset for correct hand chirality
 *
 * EULER ORDER: "YXZ" (Yaw → Pitch → Roll)
 *   - This matches typical gimbal lock avoidance for up-facing orientations
 *
 * Roll is NEGATED to match hand-3d-renderer.js:
 *   roll: -euler.roll + offset
 * This ensures both renderers respond consistently to sensor input.
 *
 * =====================================================================
 */

const THREE = window.THREE;

/**
 * Finger bone configuration
 * Each finger has: metacarpal (optional), proximal, intermediate, distal
 */
const FINGER_CONFIG = {
  thumb: {
    baseOffset: { x: -0.4, y: 0.1, z: 0.1 },
    baseRotation: { x: 0, y: 0, z: Math.PI / 6 },
    bones: [
      { name: "metacarpal", length: 0.3 },
      { name: "proximal", length: 0.25 },
      { name: "distal", length: 0.2 },
    ],
  },
  index: {
    baseOffset: { x: -0.25, y: 0.5, z: 0 },
    baseRotation: { x: 0, y: 0, z: 0 },
    bones: [
      { name: "proximal", length: 0.3 },
      { name: "intermediate", length: 0.2 },
      { name: "distal", length: 0.15 },
    ],
  },
  middle: {
    baseOffset: { x: 0, y: 0.55, z: 0 },
    baseRotation: { x: 0, y: 0, z: 0 },
    bones: [
      { name: "proximal", length: 0.35 },
      { name: "intermediate", length: 0.22 },
      { name: "distal", length: 0.15 },
    ],
  },
  ring: {
    baseOffset: { x: 0.2, y: 0.5, z: 0 },
    baseRotation: { x: 0, y: 0, z: 0 },
    bones: [
      { name: "proximal", length: 0.32 },
      { name: "intermediate", length: 0.2 },
      { name: "distal", length: 0.14 },
    ],
  },
  pinky: {
    baseOffset: { x: 0.38, y: 0.4, z: 0 },
    baseRotation: { x: 0, y: 0, z: 0 },
    bones: [
      { name: "proximal", length: 0.25 },
      { name: "intermediate", length: 0.15 },
      { name: "distal", length: 0.12 },
    ],
  },
};

/**
 * Finger curl presets (rotation in radians for each joint)
 */
const FINGER_CURL_PRESETS = {
  extended: [0, 0, 0],
  relaxed: [0.2, 0.3, 0.2],
  partial: [0.5, 0.7, 0.5],
  flexed: [1.2, 1.4, 1.0],
};

// FIX: shortest-path lerp for degrees to avoid 360/0 long-way spins.
function lerpAngleDeg(a, b, t) {
  // delta in [-180, 180)
  const delta = ((((b - a) % 360) + 540) % 360) - 180;
  return a + delta * t;
}

export class ThreeJSHandSkeleton {
  constructor(container, options = {}) {
    if (!THREE) {
      throw new Error("THREE.js not loaded. Include Three.js before this module.");
    }

    this.container = container;
    this.options = {
      width: options.width || container.clientWidth || 300,
      height: options.height || container.clientHeight || 300,
      backgroundColor: options.backgroundColor ?? 0x1a1a2e,
      boneColor: options.boneColor ?? 0x00ff88,
      jointColor: options.jointColor ?? 0xff6600,

      // Visualization toggles (all optional)
      showAxesHelper: options.showAxesHelper ?? true,
      showSkeletonHelper: options.showSkeletonHelper ?? true, // wireframe lines
      showJointSpheres: options.showJointSpheres ?? true,     // your joints
      showBoneCylinders: options.showBoneCylinders ?? true,   // your bone cylinders

      ...options,
    };

    // Orientation state (degrees)
    this.currentOrientation = { roll: 0, pitch: 0, yaw: 0 };
    this.targetOrientation = { roll: 0, pitch: 0, yaw: 0 };
    this.orientationLerpFactor = options.lerpFactor ?? 0.15;

    // Offsets (degrees)
    this.orientationOffsets = { roll: 0, pitch: 0, yaw: 0 };

    // Finger curl state (0-1 for each finger)
    this.fingerCurls = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };

    // Store bone references for manipulation
    this.bones = {};
    this.fingerBones = {};

    // Visual meshes tagged by type for toggling
    this.visualMeshes = [];

    this._initScene();
    this._createHandSkeleton();
    this._applyVisualizationVisibility(); // apply initial visibility toggles
    this._startRenderLoop();
  }

  _initScene() {
    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.options.backgroundColor);

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      50,
      this.options.width / this.options.height,
      0.1,
      100
    );
    this.camera.position.set(0, 0.5, 3);
    this.camera.lookAt(0, 0.5, 0);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.options.width, this.options.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // Lights
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(2, 3, 4);
    this.scene.add(directionalLight);

    // Axes helper
    this.axesHelper = new THREE.AxesHelper(0.3);
    this.axesHelper.position.set(-1, -0.5, 0);
    this.axesHelper.visible = !!this.options.showAxesHelper;
    this.scene.add(this.axesHelper);
  }

  _createHandSkeleton() {
    // Root group for the entire hand (rotated for orientation)
    this.handGroup = new THREE.Group();
    this.scene.add(this.handGroup);

    // Wrist bone
    const wristBone = new THREE.Bone();
    wristBone.name = "wrist";
    wristBone.position.set(0, 0, 0);
    this.bones.wrist = wristBone;
    this.wristBone = wristBone;

    // Palm bone
    const palmBone = new THREE.Bone();
    palmBone.name = "palm";
    palmBone.position.set(0, 0.3, 0);
    wristBone.add(palmBone);
    this.bones.palm = palmBone;

    // Fingers
    ["thumb", "index", "middle", "ring", "pinky"].forEach((fingerName) => {
      this._createFingerBones(palmBone, fingerName);
    });

    // Skeleton
    const allBones = this._collectAllBones(wristBone);
    this.skeleton = new THREE.Skeleton(allBones);

    // Your custom visual representation
    this._createBoneVisualization(wristBone);

    // Add wrist bone hierarchy to hand group
    this.handGroup.add(wristBone);

    // FIX: SkeletonHelper should NOT be parented under the same rotating group
    // in a way that causes apparent double transforms. Add to scene and update().
    this.skeletonHelper = new THREE.SkeletonHelper(wristBone);
    this.skeletonHelper.material.linewidth = 2;
    this.skeletonHelper.visible = !!this.options.showSkeletonHelper;
    this.scene.add(this.skeletonHelper);
  }

  _createFingerBones(palmBone, fingerName) {
    const config = FINGER_CONFIG[fingerName];
    this.fingerBones[fingerName] = [];

    let prevBone = null;

    config.bones.forEach((boneConfig, index) => {
      const bone = new THREE.Bone();
      bone.name = `${fingerName}_${boneConfig.name}`;

      if (index === 0) {
        bone.position.set(config.baseOffset.x, config.baseOffset.y, config.baseOffset.z);

        if (config.baseRotation) {
          bone.rotation.set(config.baseRotation.x, config.baseRotation.y, config.baseRotation.z);
        }

        palmBone.add(bone);
      } else {
        bone.position.set(0, config.bones[index - 1].length, 0);
        prevBone.add(bone);
      }

      this.fingerBones[fingerName].push(bone);
      this.bones[bone.name] = bone;
      prevBone = bone;
    });

    // Tip marker
    const tipBone = new THREE.Bone();
    tipBone.name = `${fingerName}_tip`;
    tipBone.position.set(0, config.bones[config.bones.length - 1].length, 0);
    prevBone.add(tipBone);
    this.bones[tipBone.name] = tipBone;
  }

  _collectAllBones(rootBone) {
    const bones = [rootBone];
    rootBone.traverse((child) => {
      if (child !== rootBone && child.isBone) bones.push(child);
    });
    return bones;
  }

  _createBoneVisualization(rootBone) {
    const jointGeometry = new THREE.SphereGeometry(0.04, 8, 8);
    const jointMaterial = new THREE.MeshPhongMaterial({ color: this.options.jointColor });

    const boneGeometry = new THREE.CylinderGeometry(0.02, 0.025, 1, 6);
    const boneMaterial = new THREE.MeshPhongMaterial({ color: this.options.boneColor });

    rootBone.traverse((bone) => {
      if (!bone.isBone) return;

      // Joint sphere
      const jointMesh = new THREE.Mesh(jointGeometry, jointMaterial);
      jointMesh.userData.vizType = "joint";
      bone.add(jointMesh);
      this.visualMeshes.push(jointMesh);

      // Cylinders to children
      bone.children.forEach((child) => {
        if (!child.isBone) return;

        const length = child.position.length();
        if (length > 0.01) {
          const cylinder = new THREE.Mesh(boneGeometry, boneMaterial);
          cylinder.userData.vizType = "bone";
          cylinder.scale.y = length;
          cylinder.position.copy(child.position).multiplyScalar(0.5);
          cylinder.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            child.position.clone().normalize()
          );
          bone.add(cylinder);
          this.visualMeshes.push(cylinder);
        }
      });
    });
  }

  /**
   * Update hand orientation from sensor fusion euler angles
   * @param {Object} euler - { roll, pitch, yaw } in degrees
   */
  updateOrientation(euler) {
    if (!euler) return;
    this.targetOrientation = {
      roll: euler.roll ?? 0,
      pitch: euler.pitch ?? 0,
      yaw: euler.yaw ?? 0,
    };
  }

  /**
   * Calibrate neutral hand offsets (degrees)
   */
  setOrientationOffsets(offsets) {
    this.orientationOffsets = {
      roll: offsets?.roll ?? 0,
      pitch: offsets?.pitch ?? 0,
      yaw: offsets?.yaw ?? 0,
    };
  }

  setFingerCurl(fingerName, amount) {
    if (Object.prototype.hasOwnProperty.call(this.fingerCurls, fingerName)) {
      this.fingerCurls[fingerName] = Math.max(0, Math.min(1, amount));
    }
  }

  setFingerCurls(curls) {
    Object.keys(curls || {}).forEach((finger) => {
      this.setFingerCurl(finger, curls[finger]);
    });
  }

  _applyFingerCurls() {
    Object.keys(this.fingerBones).forEach((fingerName) => {
      const bones = this.fingerBones[fingerName];
      const curl = this.fingerCurls[fingerName];

      const extendedAngles = FINGER_CURL_PRESETS.extended;
      const flexedAngles = FINGER_CURL_PRESETS.flexed;

      bones.forEach((bone, index) => {
        if (index < extendedAngles.length) {
          const targetAngle =
            extendedAngles[index] + (flexedAngles[index] - extendedAngles[index]) * curl;

          if (fingerName === "thumb" && index === 0) {
            bone.rotation.x = FINGER_CONFIG.thumb.baseRotation.x + targetAngle;
          } else if (index === 0 && FINGER_CONFIG[fingerName].baseRotation) {
            bone.rotation.x = targetAngle;
            bone.rotation.z = FINGER_CONFIG[fingerName].baseRotation.z;
          } else {
            bone.rotation.x = targetAngle;
          }
        }
      });
    });
  }

  _applyOrientation() {
    // Wrap-aware shortest-path lerp for all axes (degrees)
    this.currentOrientation.roll = lerpAngleDeg(
      this.currentOrientation.roll,
      this.targetOrientation.roll,
      this.orientationLerpFactor
    );
    this.currentOrientation.pitch = lerpAngleDeg(
      this.currentOrientation.pitch,
      this.targetOrientation.pitch,
      this.orientationLerpFactor
    );
    this.currentOrientation.yaw = lerpAngleDeg(
      this.currentOrientation.yaw,
      this.targetOrientation.yaw,
      this.orientationLerpFactor
    );

    const offsets = this.orientationOffsets || { roll: 0, pitch: 0, yaw: 0 };

    // Match hand-3d-renderer.js mapping for consistency:
    // - Roll is NEGATED to align sensor X axis (toward wrist) with hand model Z axis
    // - See docs/procedures/orientation-validation-protocol.md for coordinate systems
    const roll = (-this.currentOrientation.roll + offsets.roll) * (Math.PI / 180);
    const pitch = (this.currentOrientation.pitch + offsets.pitch) * (Math.PI / 180);
    const yaw = (this.currentOrientation.yaw + offsets.yaw) * (Math.PI / 180);

    this.handGroup.rotation.set(
      pitch, // X
      yaw,   // Y
      roll,  // Z
      "YXZ"
    );
  }

  _render() {
    this._applyOrientation();
    this._applyFingerCurls();

    // Update bone world matrices before rendering
    // SkeletonHelper updates automatically via onBeforeRender in Three.js r160+
    if (this.wristBone) this.wristBone.updateWorldMatrix(true, true);

    this.renderer.render(this.scene, this.camera);
  }

  _startRenderLoop() {
    this.animationId = null;
    const animate = () => {
      this.animationId = requestAnimationFrame(animate);
      this._render();
    };
    animate();
  }

  /**
   * Visibility controls
   */
  _applyVisualizationVisibility() {
    // Axes
    if (this.axesHelper) this.axesHelper.visible = !!this.options.showAxesHelper;

    // SkeletonHelper wireframe
    if (this.skeletonHelper) this.skeletonHelper.visible = !!this.options.showSkeletonHelper;

    // Custom joint spheres + bone cylinders
    this.visualMeshes.forEach((m) => {
      const type = m.userData?.vizType;
      if (type === "joint") m.visible = !!this.options.showJointSpheres;
      else if (type === "bone") m.visible = !!this.options.showBoneCylinders;
      else m.visible = true;
    });
  }

  /**
   * Toggle individual visualization layers at runtime.
   * Pass only what you want to change.
   */
  setVisualization(options = {}) {
    if (typeof options.showAxesHelper === "boolean") this.options.showAxesHelper = options.showAxesHelper;
    if (typeof options.showSkeletonHelper === "boolean") this.options.showSkeletonHelper = options.showSkeletonHelper;
    if (typeof options.showJointSpheres === "boolean") this.options.showJointSpheres = options.showJointSpheres;
    if (typeof options.showBoneCylinders === "boolean") this.options.showBoneCylinders = options.showBoneCylinders;
    this._applyVisualizationVisibility();
  }

  resetOrientation() {
    this.targetOrientation = { roll: 0, pitch: 0, yaw: 0 };
    this.currentOrientation = { roll: 0, pitch: 0, yaw: 0 };
  }

  resize(width, height) {
    this.options.width = width;
    this.options.height = height;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  getOrientation() {
    return { ...this.currentOrientation };
  }

  getBone(name) {
    return this.bones[name];
  }

  dispose() {
    if (this.animationId) cancelAnimationFrame(this.animationId);

    // Dispose geometries/materials from custom meshes
    this.visualMeshes.forEach((mesh) => {
      if (mesh.geometry) mesh.geometry.dispose();
      if (mesh.material) mesh.material.dispose();
    });

    // SkeletonHelper
    if (this.skeletonHelper) {
      if (this.skeletonHelper.parent) this.skeletonHelper.parent.remove(this.skeletonHelper);
      this.skeletonHelper.dispose?.();
      this.skeletonHelper = null;
    }

    // Axes helper
    if (this.axesHelper) {
      if (this.axesHelper.parent) this.axesHelper.parent.remove(this.axesHelper);
      this.axesHelper = null;
    }

    // Renderer
    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement?.parentNode === this.container) {
        this.container.removeChild(this.renderer.domElement);
      }
    }
  }
}

export default ThreeJSHandSkeleton;

/**
 * Example toggles:
 *
 *   hand.setVisualization({ showSkeletonHelper: true, showJointSpheres: true, showBoneCylinders: true });
 *   hand.setVisualization({ showSkeletonHelper: false }); // only custom
 *   hand.setVisualization({ showJointSpheres: false, showBoneCylinders: false }); // only helper lines
 */
