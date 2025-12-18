/**
 * Three.js Hand Skeleton Module
 *
 * Provides 3D skeletal hand visualization driven by IMU sensor orientation.
 *
 * @module shared/threejs-hand-skeleton
 */

import type { EulerAngles } from '@core/types';

// ===== Type Definitions =====

declare const THREE: any;

export interface HandSkeletonOptions {
  width?: number;
  height?: number;
  backgroundColor?: number;
  boneColor?: number;
  jointColor?: number;
  showAxesHelper?: boolean;
  showSkeletonHelper?: boolean;
  showJointSpheres?: boolean;
  showBoneCylinders?: boolean;
  lerpFactor?: number;
  negateRoll?: boolean;
  negatePitch?: boolean;
  negateYaw?: boolean;
  handedness?: 'left' | 'right';
}

export interface OrientationOffsets {
  roll: number;
  pitch: number;
  yaw: number;
}

export interface AxisSigns {
  negateRoll: boolean;
  negatePitch: boolean;
  negateYaw: boolean;
}

export interface FingerCurls {
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  pinky: number;
}

export interface VisualizationOptions {
  showAxesHelper?: boolean;
  showSkeletonHelper?: boolean;
  showJointSpheres?: boolean;
  showBoneCylinders?: boolean;
}

interface BoneConfig {
  name: string;
  length: number;
}

interface FingerConfig {
  baseOffset: { x: number; y: number; z: number };
  baseRotation: { x: number; y: number; z: number };
  bones: BoneConfig[];
}

// ===== Configuration =====

const FINGER_CONFIG: Record<string, FingerConfig> = {
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

const FINGER_CURL_PRESETS: Record<string, number[]> = {
  extended: [0, 0, 0],
  relaxed: [0.2, 0.3, 0.2],
  partial: [0.5, 0.7, 0.5],
  flexed: [1.2, 1.4, 1.0],
};

function lerpAngleDeg(a: number, b: number, t: number): number {
  const delta = ((((b - a) % 360) + 540) % 360) - 180;
  return a + delta * t;
}

// ===== ThreeJSHandSkeleton Class =====

export class ThreeJSHandSkeleton {
  private container: HTMLElement;
  private options: Required<HandSkeletonOptions>;
  private currentOrientation: OrientationOffsets;
  private targetOrientation: OrientationOffsets;
  private orientationLerpFactor: number;
  private orientationOffsets: OrientationOffsets;
  private axisSigns: AxisSigns;
  private handedness: 'left' | 'right';
  private fingerCurls: FingerCurls;
  private bones: Record<string, any>;
  private fingerBones: Record<string, any[]>;
  private visualMeshes: any[];
  private scene: any;
  private camera: any;
  private renderer: any;
  private axesHelper: any;
  private handGroup: any;
  private wristBone: any;
  private skeleton: any;
  private skeletonHelper: any;
  private animationId: number | null;

  constructor(container: HTMLElement, options: HandSkeletonOptions = {}) {
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
      showAxesHelper: options.showAxesHelper ?? true,
      showSkeletonHelper: options.showSkeletonHelper ?? true,
      showJointSpheres: options.showJointSpheres ?? true,
      showBoneCylinders: options.showBoneCylinders ?? true,
      lerpFactor: options.lerpFactor ?? 0.15,
      negateRoll: options.negateRoll ?? false,
      negatePitch: options.negatePitch ?? true,
      negateYaw: options.negateYaw ?? false,
      handedness: options.handedness ?? 'right',
    };

    this.currentOrientation = { roll: 0, pitch: 0, yaw: 0 };
    this.targetOrientation = { roll: 0, pitch: 0, yaw: 0 };
    this.orientationLerpFactor = this.options.lerpFactor;
    this.orientationOffsets = { roll: 0, pitch: 0, yaw: 0 };

    this.axisSigns = {
      negateRoll: this.options.negateRoll,
      negatePitch: this.options.negatePitch,
      negateYaw: this.options.negateYaw
    };

    this.handedness = this.options.handedness;
    this.fingerCurls = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
    this.bones = {};
    this.fingerBones = {};
    this.visualMeshes = [];
    this.animationId = null;

    this._initScene();
    this._createHandSkeleton();
    this._applyVisualizationVisibility();
    this._applyHandedness();
    this._startRenderLoop();
  }

  private _initScene(): void {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.options.backgroundColor);

    this.camera = new THREE.PerspectiveCamera(
      50,
      this.options.width / this.options.height,
      0.1,
      100
    );
    this.camera.position.set(0, 0.5, 3);
    this.camera.lookAt(0, 0.5, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.options.width, this.options.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(2, 3, 4);
    this.scene.add(directionalLight);

    this.axesHelper = new THREE.AxesHelper(0.3);
    this.axesHelper.position.set(-1, -0.5, 0);
    this.axesHelper.visible = !!this.options.showAxesHelper;
    this.scene.add(this.axesHelper);
  }

  private _createHandSkeleton(): void {
    this.handGroup = new THREE.Group();
    this.scene.add(this.handGroup);

    const wristBone = new THREE.Bone();
    wristBone.name = "wrist";
    wristBone.position.set(0, 0, 0);
    this.bones.wrist = wristBone;
    this.wristBone = wristBone;

    const palmBone = new THREE.Bone();
    palmBone.name = "palm";
    palmBone.position.set(0, 0.3, 0);
    wristBone.add(palmBone);
    this.bones.palm = palmBone;

    ["thumb", "index", "middle", "ring", "pinky"].forEach((fingerName) => {
      this._createFingerBones(palmBone, fingerName);
    });

    const allBones = this._collectAllBones(wristBone);
    this.skeleton = new THREE.Skeleton(allBones);

    this._createBoneVisualization(wristBone);
    this.handGroup.add(wristBone);

    this.skeletonHelper = new THREE.SkeletonHelper(wristBone);
    this.skeletonHelper.material.linewidth = 2;
    this.skeletonHelper.visible = !!this.options.showSkeletonHelper;
    this.scene.add(this.skeletonHelper);
  }

  private _createFingerBones(palmBone: any, fingerName: string): void {
    const config = FINGER_CONFIG[fingerName];
    this.fingerBones[fingerName] = [];

    let prevBone: any = null;

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

    const tipBone = new THREE.Bone();
    tipBone.name = `${fingerName}_tip`;
    tipBone.position.set(0, config.bones[config.bones.length - 1].length, 0);
    prevBone.add(tipBone);
    this.bones[tipBone.name] = tipBone;
  }

  private _collectAllBones(rootBone: any): any[] {
    const bones = [rootBone];
    rootBone.traverse((child: any) => {
      if (child !== rootBone && child.isBone) bones.push(child);
    });
    return bones;
  }

  private _createBoneVisualization(rootBone: any): void {
    const jointGeometry = new THREE.SphereGeometry(0.04, 8, 8);
    const jointMaterial = new THREE.MeshPhongMaterial({ color: this.options.jointColor });

    const boneGeometry = new THREE.CylinderGeometry(0.02, 0.025, 1, 6);
    const boneMaterial = new THREE.MeshPhongMaterial({ color: this.options.boneColor });

    rootBone.traverse((bone: any) => {
      if (!bone.isBone) return;

      const jointMesh = new THREE.Mesh(jointGeometry, jointMaterial);
      jointMesh.userData.vizType = "joint";
      bone.add(jointMesh);
      this.visualMeshes.push(jointMesh);

      bone.children.forEach((child: any) => {
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

  updateOrientation(euler: EulerAngles | null): void {
    if (!euler) return;
    this.targetOrientation = {
      roll: euler.roll ?? 0,
      pitch: euler.pitch ?? 0,
      yaw: euler.yaw ?? 0,
    };
  }

  setOrientationOffsets(offsets: Partial<OrientationOffsets> | null): void {
    this.orientationOffsets = {
      roll: offsets?.roll ?? 0,
      pitch: offsets?.pitch ?? 0,
      yaw: offsets?.yaw ?? 0,
    };
  }

  setFingerCurl(fingerName: keyof FingerCurls, amount: number): void {
    if (Object.prototype.hasOwnProperty.call(this.fingerCurls, fingerName)) {
      this.fingerCurls[fingerName] = Math.max(0, Math.min(1, amount));
    }
  }

  setFingerCurls(curls: Partial<FingerCurls> | null): void {
    Object.keys(curls || {}).forEach((finger) => {
      this.setFingerCurl(finger as keyof FingerCurls, (curls as any)[finger]);
    });
  }

  getRenderState(): { x: number; y: number; z: number; order: string } {
    if (this.handGroup && this.handGroup.rotation) {
      const rot = this.handGroup.rotation;
      return { x: rot.x, y: rot.y, z: rot.z, order: rot.order };
    }
    return { x: 0, y: 0, z: 0, order: 'YXZ' };
  }

  private _applyFingerCurls(): void {
    Object.keys(this.fingerBones).forEach((fingerName) => {
      const bones = this.fingerBones[fingerName];
      const curl = this.fingerCurls[fingerName as keyof FingerCurls];

      const extendedAngles = FINGER_CURL_PRESETS.extended;
      const flexedAngles = FINGER_CURL_PRESETS.flexed;

      bones.forEach((bone: any, index: number) => {
        if (index < extendedAngles.length) {
          const targetAngle =
            extendedAngles[index] + (flexedAngles[index] - extendedAngles[index]) * curl;

          if (fingerName === "thumb") {
            if (index === 0) {
              bone.rotation.x = FINGER_CONFIG.thumb.baseRotation.x;
              bone.rotation.z = FINGER_CONFIG.thumb.baseRotation.z - targetAngle;
            } else {
              bone.rotation.z = 0 - targetAngle;
            }
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

  private _applyOrientation(): void {
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
    const signs = this.axisSigns || { negateRoll: false, negatePitch: true, negateYaw: false };

    const sensorRoll = this.currentOrientation.roll;
    const sensorPitch = this.currentOrientation.pitch;
    const sensorYaw = this.currentOrientation.yaw;

    const pitch = ((signs.negatePitch ? -sensorPitch : sensorPitch) + offsets.pitch) * (Math.PI / 180);
    const yaw = (((signs.negateYaw ? 1 : -1) * sensorRoll) + offsets.yaw) * (Math.PI / 180);
    const roll = ((signs.negateRoll ? -sensorYaw : sensorYaw) + offsets.roll) * (Math.PI / 180);

    this.handGroup.rotation.set(pitch, yaw, roll, "YXZ");
  }

  setAxisSigns(signs: Partial<AxisSigns>): void {
    if (signs.negateRoll !== undefined) this.axisSigns.negateRoll = signs.negateRoll;
    if (signs.negatePitch !== undefined) this.axisSigns.negatePitch = signs.negatePitch;
    if (signs.negateYaw !== undefined) this.axisSigns.negateYaw = signs.negateYaw;
  }

  getAxisSigns(): AxisSigns {
    return { ...this.axisSigns };
  }

  private _render(): void {
    this._applyOrientation();
    this._applyFingerCurls();

    if (this.wristBone) this.wristBone.updateWorldMatrix(true, true);

    this.renderer.render(this.scene, this.camera);
  }

  private _startRenderLoop(): void {
    this.animationId = null;
    const animate = () => {
      this.animationId = requestAnimationFrame(animate);
      this._render();
    };
    animate();
  }

  private _applyHandedness(): void {
    if (!this.handGroup) return;
    this.handGroup.scale.x = this.handedness === 'left' ? 1 : -1;
  }

  setHandedness(hand: 'left' | 'right'): void {
    if (hand !== 'left' && hand !== 'right') {
      console.warn('[ThreeHand] Invalid handedness:', hand);
      return;
    }

    this.handedness = hand;
    this._applyHandedness();
  }

  getHandedness(): 'left' | 'right' {
    return this.handedness;
  }

  private _applyVisualizationVisibility(): void {
    if (this.axesHelper) this.axesHelper.visible = !!this.options.showAxesHelper;
    if (this.skeletonHelper) this.skeletonHelper.visible = !!this.options.showSkeletonHelper;

    this.visualMeshes.forEach((m) => {
      const type = m.userData?.vizType;
      if (type === "joint") m.visible = !!this.options.showJointSpheres;
      else if (type === "bone") m.visible = !!this.options.showBoneCylinders;
      else m.visible = true;
    });
  }

  setVisualization(options: VisualizationOptions = {}): void {
    if (typeof options.showAxesHelper === "boolean") this.options.showAxesHelper = options.showAxesHelper;
    if (typeof options.showSkeletonHelper === "boolean") this.options.showSkeletonHelper = options.showSkeletonHelper;
    if (typeof options.showJointSpheres === "boolean") this.options.showJointSpheres = options.showJointSpheres;
    if (typeof options.showBoneCylinders === "boolean") this.options.showBoneCylinders = options.showBoneCylinders;
    this._applyVisualizationVisibility();
  }

  resetOrientation(): void {
    this.targetOrientation = { roll: 0, pitch: 0, yaw: 0 };
    this.currentOrientation = { roll: 0, pitch: 0, yaw: 0 };
  }

  resize(width: number, height: number): void {
    this.options.width = width;
    this.options.height = height;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  getOrientation(): OrientationOffsets {
    return { ...this.currentOrientation };
  }

  getBone(name: string): any {
    return this.bones[name];
  }

  dispose(): void {
    if (this.animationId) cancelAnimationFrame(this.animationId);

    this.visualMeshes.forEach((mesh) => {
      if (mesh.geometry) mesh.geometry.dispose();
      if (mesh.material) mesh.material.dispose();
    });

    if (this.skeletonHelper) {
      if (this.skeletonHelper.parent) this.skeletonHelper.parent.remove(this.skeletonHelper);
      this.skeletonHelper.dispose?.();
      this.skeletonHelper = null;
    }

    if (this.axesHelper) {
      if (this.axesHelper.parent) this.axesHelper.parent.remove(this.axesHelper);
      this.axesHelper = null;
    }

    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement?.parentNode === this.container) {
        this.container.removeChild(this.renderer.domElement);
      }
    }
  }
}

export default ThreeJSHandSkeleton;
