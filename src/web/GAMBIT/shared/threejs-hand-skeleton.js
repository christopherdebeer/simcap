/**
 * Three.js Hand Skeleton Module
 *
 * A barebones skeletal hand for mapping sensor orientation to a 3D hand model.
 * Built on THREE.Bone/Skeleton for easy iteration.
 *
 * Coordinate System:
 * - Fingers extend in +Y direction
 * - Palm faces +Z (towards viewer when neutral)
 * - Thumb on -X side (right hand)
 *
 * Usage:
 *   const hand = new ThreeJSHandSkeleton(containerElement);
 *   hand.updateOrientation({ roll, pitch, yaw }); // euler angles in degrees
 *   hand.dispose();
 */

// Wait for THREE to be available (loaded via CDN)
const THREE = window.THREE;

/**
 * Finger bone configuration
 * Each finger has: metacarpal (optional), proximal, intermediate, distal
 */
const FINGER_CONFIG = {
    thumb: {
        // Thumb is offset and rotated
        baseOffset: { x: -0.4, y: 0.1, z: 0.1 },
        baseRotation: { x: 0, y: 0, z: Math.PI / 6 }, // Angled outward
        bones: [
            { name: 'metacarpal', length: 0.3 },
            { name: 'proximal', length: 0.25 },
            { name: 'distal', length: 0.2 }
        ]
    },
    index: {
        baseOffset: { x: -0.25, y: 0.5, z: 0 },
        baseRotation: { x: 0, y: 0, z: 0 },
        bones: [
            { name: 'proximal', length: 0.3 },
            { name: 'intermediate', length: 0.2 },
            { name: 'distal', length: 0.15 }
        ]
    },
    middle: {
        baseOffset: { x: 0, y: 0.55, z: 0 },
        baseRotation: { x: 0, y: 0, z: 0 },
        bones: [
            { name: 'proximal', length: 0.35 },
            { name: 'intermediate', length: 0.22 },
            { name: 'distal', length: 0.15 }
        ]
    },
    ring: {
        baseOffset: { x: 0.2, y: 0.5, z: 0 },
        baseRotation: { x: 0, y: 0, z: 0 },
        bones: [
            { name: 'proximal', length: 0.32 },
            { name: 'intermediate', length: 0.2 },
            { name: 'distal', length: 0.14 }
        ]
    },
    pinky: {
        baseOffset: { x: 0.38, y: 0.4, z: 0 },
        baseRotation: { x: 0, y: 0, z: 0 },
        bones: [
            { name: 'proximal', length: 0.25 },
            { name: 'intermediate', length: 0.15 },
            { name: 'distal', length: 0.12 }
        ]
    }
};

/**
 * Finger curl presets (rotation in radians for each joint)
 * Will be used later for finger pose mapping
 */
const FINGER_CURL_PRESETS = {
    extended: [0, 0, 0],           // All joints straight
    relaxed: [0.2, 0.3, 0.2],      // Slight curl
    partial: [0.5, 0.7, 0.5],      // Half curled
    flexed: [1.2, 1.4, 1.0]        // Fully curled (fist)
};

export class ThreeJSHandSkeleton {
    constructor(container, options = {}) {
        if (!THREE) {
            throw new Error('THREE.js not loaded. Include Three.js before this module.');
        }

        this.container = container;
        this.options = {
            width: options.width || container.clientWidth || 300,
            height: options.height || container.clientHeight || 300,
            backgroundColor: options.backgroundColor || 0x1a1a2e,
            boneColor: options.boneColor || 0x00ff88,
            jointColor: options.jointColor || 0xff6600,
            ...options
        };

        // Orientation state
        this.currentOrientation = { roll: 0, pitch: 0, yaw: 0 };
        this.targetOrientation = { roll: 0, pitch: 0, yaw: 0 };
        this.orientationLerpFactor = options.lerpFactor || 0.15;

        // Finger curl state (0-1 for each finger)
        this.fingerCurls = {
            thumb: 0,
            index: 0,
            middle: 0,
            ring: 0,
            pinky: 0
        };

        // Store bone references for manipulation
        this.bones = {};
        this.fingerBones = {};

        this._initScene();
        this._createHandSkeleton();
        this._startRenderLoop();
    }

    _initScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);

        // Camera - positioned to view hand from front
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
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(2, 3, 4);
        this.scene.add(directionalLight);

        // Add coordinate axes helper (small, at origin)
        const axesHelper = new THREE.AxesHelper(0.3);
        axesHelper.position.set(-1, -0.5, 0);
        this.scene.add(axesHelper);
    }

    _createHandSkeleton() {
        // Root group for the entire hand (this is what we rotate for orientation)
        this.handGroup = new THREE.Group();
        this.scene.add(this.handGroup);

        // Wrist bone (root of skeleton)
        const wristBone = new THREE.Bone();
        wristBone.name = 'wrist';
        wristBone.position.set(0, 0, 0);
        this.bones.wrist = wristBone;

        // Palm bone (attached to wrist)
        const palmBone = new THREE.Bone();
        palmBone.name = 'palm';
        palmBone.position.set(0, 0.3, 0);
        wristBone.add(palmBone);
        this.bones.palm = palmBone;

        // Create finger bones
        const fingerNames = ['thumb', 'index', 'middle', 'ring', 'pinky'];
        fingerNames.forEach(fingerName => {
            this._createFingerBones(palmBone, fingerName);
        });

        // Create skeleton
        const allBones = this._collectAllBones(wristBone);
        this.skeleton = new THREE.Skeleton(allBones);

        // Create visual representation of bones
        this._createBoneVisualization(wristBone);

        // Add wrist bone to hand group
        this.handGroup.add(wristBone);

        // Create skeleton helper (shows bone connections)
        this.skeletonHelper = new THREE.SkeletonHelper(wristBone);
        this.skeletonHelper.material.linewidth = 2;
        this.handGroup.add(this.skeletonHelper);
    }

    _createFingerBones(palmBone, fingerName) {
        const config = FINGER_CONFIG[fingerName];
        this.fingerBones[fingerName] = [];

        // Create base position bone (attached to palm)
        let parentBone = palmBone;
        let prevBone = null;

        config.bones.forEach((boneConfig, index) => {
            const bone = new THREE.Bone();
            bone.name = `${fingerName}_${boneConfig.name}`;

            if (index === 0) {
                // First bone - position relative to palm
                bone.position.set(
                    config.baseOffset.x,
                    config.baseOffset.y,
                    config.baseOffset.z
                );
                // Apply base rotation for thumb
                if (config.baseRotation) {
                    bone.rotation.set(
                        config.baseRotation.x,
                        config.baseRotation.y,
                        config.baseRotation.z
                    );
                }
                palmBone.add(bone);
            } else {
                // Subsequent bones - positioned at end of previous bone
                bone.position.set(0, config.bones[index - 1].length, 0);
                prevBone.add(bone);
            }

            this.fingerBones[fingerName].push(bone);
            this.bones[bone.name] = bone;
            prevBone = bone;
        });

        // Add tip marker
        const tipBone = new THREE.Bone();
        tipBone.name = `${fingerName}_tip`;
        tipBone.position.set(0, config.bones[config.bones.length - 1].length, 0);
        prevBone.add(tipBone);
        this.bones[tipBone.name] = tipBone;
    }

    _collectAllBones(rootBone) {
        const bones = [rootBone];
        rootBone.traverse(child => {
            if (child !== rootBone && child.isBone) {
                bones.push(child);
            }
        });
        return bones;
    }

    _createBoneVisualization(rootBone) {
        // Create visual spheres for joints and cylinders for bones
        const jointGeometry = new THREE.SphereGeometry(0.04, 8, 8);
        const jointMaterial = new THREE.MeshPhongMaterial({ color: this.options.jointColor });

        const boneGeometry = new THREE.CylinderGeometry(0.02, 0.025, 1, 6);
        const boneMaterial = new THREE.MeshPhongMaterial({ color: this.options.boneColor });

        this.visualMeshes = [];

        rootBone.traverse(bone => {
            if (!bone.isBone) return;

            // Joint sphere at bone position
            const jointMesh = new THREE.Mesh(jointGeometry, jointMaterial);
            bone.add(jointMesh);
            this.visualMeshes.push(jointMesh);

            // Create bone cylinder to child
            bone.children.forEach(child => {
                if (!child.isBone) return;

                const length = child.position.length();
                if (length > 0.01) {
                    const cylinder = new THREE.Mesh(boneGeometry, boneMaterial);
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

        // Store target orientation (will lerp towards this)
        this.targetOrientation = {
            roll: euler.roll || 0,
            pitch: euler.pitch || 0,
            yaw: euler.yaw || 0
        };
    }

    /**
     * Set the orientation mapping offsets
     * Used to calibrate the neutral hand position
     * @param {Object} offsets - { roll, pitch, yaw } offset in degrees
     */
    setOrientationOffsets(offsets) {
        this.orientationOffsets = {
            roll: offsets.roll || 0,
            pitch: offsets.pitch || 0,
            yaw: offsets.yaw || 0
        };
    }

    /**
     * Update finger curl amount
     * @param {string} fingerName - 'thumb', 'index', 'middle', 'ring', 'pinky'
     * @param {number} amount - 0 (extended) to 1 (fully flexed)
     */
    setFingerCurl(fingerName, amount) {
        if (this.fingerCurls.hasOwnProperty(fingerName)) {
            this.fingerCurls[fingerName] = Math.max(0, Math.min(1, amount));
        }
    }

    /**
     * Set all finger curls at once
     * @param {Object} curls - { thumb, index, middle, ring, pinky } each 0-1
     */
    setFingerCurls(curls) {
        Object.keys(curls).forEach(finger => {
            this.setFingerCurl(finger, curls[finger]);
        });
    }

    _applyFingerCurls() {
        // Apply curl to each finger
        Object.keys(this.fingerBones).forEach(fingerName => {
            const bones = this.fingerBones[fingerName];
            const curl = this.fingerCurls[fingerName];

            // Interpolate between extended and flexed
            const extendedAngles = FINGER_CURL_PRESETS.extended;
            const flexedAngles = FINGER_CURL_PRESETS.flexed;

            bones.forEach((bone, index) => {
                if (index < extendedAngles.length) {
                    const targetAngle = extendedAngles[index] +
                        (flexedAngles[index] - extendedAngles[index]) * curl;

                    // Fingers curl around X axis (perpendicular to finger direction)
                    // Don't override base rotation for thumb
                    if (fingerName === 'thumb' && index === 0) {
                        // Keep thumb base rotation, only add curl
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
        // Lerp current orientation towards target
        const lerp = (a, b, t) => a + (b - a) * t;

        this.currentOrientation.roll = lerp(
            this.currentOrientation.roll,
            this.targetOrientation.roll,
            this.orientationLerpFactor
        );
        this.currentOrientation.pitch = lerp(
            this.currentOrientation.pitch,
            this.targetOrientation.pitch,
            this.orientationLerpFactor
        );
        this.currentOrientation.yaw = lerp(
            this.currentOrientation.yaw,
            this.targetOrientation.yaw,
            this.orientationLerpFactor
        );

        // Apply offsets
        const offsets = this.orientationOffsets || { roll: 0, pitch: 0, yaw: 0 };

        // Convert to radians
        const roll = (this.currentOrientation.roll + offsets.roll) * Math.PI / 180;
        const pitch = (this.currentOrientation.pitch + offsets.pitch) * Math.PI / 180;
        const yaw = (this.currentOrientation.yaw + offsets.yaw) * Math.PI / 180;

        // Apply rotation to hand group
        // Mapping from sensor frame to Three.js frame:
        // - Sensor: Z up, Y forward, X right
        // - Three.js: Y up, -Z forward, X right
        //
        // This mapping may need adjustment based on how the sensor is mounted
        // and the desired neutral hand position
        this.handGroup.rotation.set(
            pitch,   // X rotation = pitch (tilt forward/back)
            yaw,     // Y rotation = yaw (rotate left/right)
            roll,    // Z rotation = roll (rotate around arm axis)
            'YXZ'    // Apply yaw first, then pitch, then roll
        );
    }

    _render() {
        this._applyOrientation();
        this._applyFingerCurls();
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
     * Reset orientation to neutral position
     */
    resetOrientation() {
        this.targetOrientation = { roll: 0, pitch: 0, yaw: 0 };
        this.currentOrientation = { roll: 0, pitch: 0, yaw: 0 };
    }

    /**
     * Resize the renderer
     */
    resize(width, height) {
        this.options.width = width;
        this.options.height = height;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    /**
     * Get current orientation values
     */
    getOrientation() {
        return { ...this.currentOrientation };
    }

    /**
     * Get direct access to bones for advanced manipulation
     */
    getBone(name) {
        return this.bones[name];
    }

    /**
     * Clean up resources
     */
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        // Dispose geometries and materials
        this.visualMeshes.forEach(mesh => {
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) mesh.material.dispose();
        });

        if (this.skeletonHelper) {
            this.skeletonHelper.dispose();
        }

        if (this.renderer) {
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }
    }
}

// Export for use as ES module
export default ThreeJSHandSkeleton;
