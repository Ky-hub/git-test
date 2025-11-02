import numpy as np
import json 

ctrl_expressions = [
    "CTRL_expressions_browDownL",
    "CTRL_expressions_browDownR",
    "CTRL_expressions_browLateralL",
    "CTRL_expressions_browLateralR",
    "CTRL_expressions_browRaiseInL",
    "CTRL_expressions_browRaiseInR",
    "CTRL_expressions_browRaiseOuterL",
    "CTRL_expressions_browRaiseOuterR",
    "CTRL_expressions_eyeBlinkL",
    "CTRL_expressions_eyeBlinkR",
    "CTRL_expressions_eyeWidenL",
    "CTRL_expressions_eyeWidenR",
    "CTRL_expressions_eyeSquintInnerL",
    "CTRL_expressions_eyeSquintInnerR",
    "CTRL_expressions_eyeCheekRaiseL",
    "CTRL_expressions_eyeCheekRaiseR",
    "CTRL_expressions_eyeFaceScrunchL",
    "CTRL_expressions_eyeFaceScrunchR",
    "CTRL_expressions_eyeLookUpL",
    "CTRL_expressions_eyeLookUpR",
    "CTRL_expressions_eyeLookDownL",
    "CTRL_expressions_eyeLookDownR",
    "CTRL_expressions_eyeLookLeftL",
    "CTRL_expressions_eyeLookLeftR",
    "CTRL_expressions_eyeLookRightL",
    "CTRL_expressions_eyeLookRightR",
    "CTRL_expressions_noseWrinkleL",
    "CTRL_expressions_noseWrinkleR",
    "CTRL_expressions_noseNostrilDepressL",
    "CTRL_expressions_noseNostrilDepressR",
    "CTRL_expressions_noseNostrilDilateL",
    "CTRL_expressions_noseNostrilDilateR",
    "CTRL_expressions_noseNostrilCompressL",
    "CTRL_expressions_noseNostrilCompressR",
    "CTRL_expressions_noseNasolabialDeepenL",
    "CTRL_expressions_noseNasolabialDeepenR",
    "CTRL_expressions_mouthCheekBlowL",
    "CTRL_expressions_mouthCheekBlowR",
    "CTRL_expressions_mouthLeft",
    "CTRL_expressions_mouthRight",
    "CTRL_expressions_mouthUpperLipRaiseL",
    "CTRL_expressions_mouthUpperLipRaiseR",
    "CTRL_expressions_mouthLowerLipDepressL",
    "CTRL_expressions_mouthLowerLipDepressR",
    "CTRL_expressions_mouthCornerPullL",
    "CTRL_expressions_mouthCornerPullR",
    "CTRL_expressions_mouthStretchL",
    "CTRL_expressions_mouthStretchR",
    "CTRL_expressions_mouthDimpleL",
    "CTRL_expressions_mouthDimpleR",
    "CTRL_expressions_mouthCornerDepressL",
    "CTRL_expressions_mouthCornerDepressR",
    "CTRL_expressions_mouthLipsPurseUL",
    "CTRL_expressions_mouthLipsPurseUR",
    "CTRL_expressions_mouthLipsPurseDL",
    "CTRL_expressions_mouthLipsPurseDR",
    "CTRL_expressions_mouthLipsTowardsUL",
    "CTRL_expressions_mouthLipsTowardsUR",
    "CTRL_expressions_mouthLipsTowardsDL",
    "CTRL_expressions_mouthLipsTowardsDR",
    "CTRL_expressions_mouthFunnelUL",
    "CTRL_expressions_mouthFunnelUR",
    "CTRL_expressions_mouthFunnelDL",
    "CTRL_expressions_mouthFunnelDR",
    "CTRL_expressions_mouthLipsTogetherUL",
    "CTRL_expressions_mouthLipsTogetherUR",
    "CTRL_expressions_mouthLipsTogetherDL",
    "CTRL_expressions_mouthLipsTogetherDR",
    "CTRL_expressions_mouthUpperLipBiteL",
    "CTRL_expressions_mouthUpperLipBiteR",
    "CTRL_expressions_mouthLowerLipBiteL",
    "CTRL_expressions_mouthLowerLipBiteR",
    "CTRL_expressions_mouthLipsTightenUL",
    "CTRL_expressions_mouthLipsTightenUR",
    "CTRL_expressions_mouthLipsTightenDL",
    "CTRL_expressions_mouthLipsTightenDR",
    "CTRL_expressions_mouthLipsPressL",
    "CTRL_expressions_mouthLipsPressR",
    "CTRL_expressions_mouthSharpCornerPullL",
    "CTRL_expressions_mouthSharpCornerPullR",
    "CTRL_expressions_mouthStickyUC",
    "CTRL_expressions_mouthStickyUINL",
    "CTRL_expressions_mouthStickyUINR",
    "CTRL_expressions_mouthStickyUOUTL",
    "CTRL_expressions_mouthStickyUOUTR",
    "CTRL_expressions_mouthStickyDC",
    "CTRL_expressions_mouthStickyDINL",
    "CTRL_expressions_mouthStickyDINR",
    "CTRL_expressions_mouthStickyDOUTL",
    "CTRL_expressions_mouthStickyDOUTR",
    "CTRL_expressions_mouthLipsPushUL",
    "CTRL_expressions_mouthLipsPushUR",
    "CTRL_expressions_mouthLipsPushDL",
    "CTRL_expressions_mouthLipsPushDR",
    "CTRL_expressions_mouthLipsPullUL",
    "CTRL_expressions_mouthLipsPullUR",
    "CTRL_expressions_mouthLipsPullDL",
    "CTRL_expressions_mouthLipsPullDR",
    "CTRL_expressions_mouthLipsThinUL",
    "CTRL_expressions_mouthLipsThinUR",
    "CTRL_expressions_mouthLipsThinDL",
    "CTRL_expressions_mouthLipsThinDR",
    "CTRL_expressions_mouthLipsThickUL",
    "CTRL_expressions_mouthLipsThickUR",
    "CTRL_expressions_mouthLipsThickDL",
    "CTRL_expressions_mouthLipsThickDR",
    "CTRL_expressions_mouthCornerSharpenUL",
    "CTRL_expressions_mouthCornerSharpenUR",
    "CTRL_expressions_mouthCornerSharpenDL",
    "CTRL_expressions_mouthCornerSharpenDR",
    "CTRL_expressions_mouthCornerRounderUL",
    "CTRL_expressions_mouthCornerRounderUR",
    "CTRL_expressions_mouthCornerRounderDL",
    "CTRL_expressions_mouthCornerRounderDR",
    "CTRL_expressions_mouthUpperLipShiftLeft",
    "CTRL_expressions_mouthLowerLipShiftLeft",
    "CTRL_expressions_mouthLowerLipShiftRight",
    "CTRL_expressions_mouthUpperLipRollInL",
    "CTRL_expressions_mouthUpperLipRollInR",
    "CTRL_expressions_mouthUpperLipRollOutL",
    "CTRL_expressions_mouthUpperLipRollOutR",
    "CTRL_expressions_mouthLowerLipRollInL",
    "CTRL_expressions_mouthLowerLipRollInR",
    "CTRL_expressions_mouthLowerLipRollOutL",
    "CTRL_expressions_mouthLowerLipRollOutR",
    "CTRL_expressions_mouthCornerUpL",
    "CTRL_expressions_mouthCornerUpR",
    "CTRL_expressions_mouthCornerDownL",
    "CTRL_expressions_mouthCornerDownR",
    "CTRL_expressions_jawOpen",
    "CTRL_expressions_jawLeft",
    "CTRL_expressions_jawRight",
    "CTRL_expressions_jawBack",
    "CTRL_expressions_jawChinRaiseDL",
    "CTRL_expressions_jawChinRaiseDR",
    "CTRL_expressions_jawOpenExtreme"
]

bone_dict = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6, "left_ankle": 7,
    "right_ankle": 8, "spine3": 9, "left_foot": 10, "right_foot": 11,
    "neck": 12, "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18,
    "right_elbow": 19, "left_wrist": 20, "right_wrist": 21,
    "jaw": 22, "left_eye_smplx": 23, "right_eye_smplx": 24,
    "left_index1": 25, "left_index2": 26, "left_index3": 27,
    "left_middle1": 28, "left_middle2": 29, "left_middle3": 30,
    "left_pinky1": 31, "left_pinky2": 32, "left_pinky3": 33,
    "left_ring1": 34, "left_ring2": 35, "left_ring3": 36,
    "left_thumb1": 37, "left_thumb2": 38, "left_thumb3": 39,
    "right_index1": 40, "right_index2": 41, "right_index3": 42,
    "right_middle1": 43, "right_middle2": 44, "right_middle3": 45,
    "right_pinky1": 46, "right_pinky2": 47, "right_pinky3": 48,
    "right_ring1": 49, "right_ring2": 50, "right_ring3": 51,
    "right_thumb1": 52, "right_thumb2": 53, "right_thumb3": 54
}

ue_to_arkit_dict = {
    # 眉毛
    "CTRL_expressions_browDownL": "browDownLeft",
    "CTRL_expressions_browDownR": "browDownRight",
    "CTRL_expressions_browLateralL": "browInnerUp",
    "CTRL_expressions_browLateralR": "browInnerUp",
    "CTRL_expressions_browRaiseInL": "browInnerUp",
    "CTRL_expressions_browRaiseInR": "browInnerUp",
    "CTRL_expressions_browRaiseOuterL": "browOuterUpLeft",
    "CTRL_expressions_browRaiseOuterR": "browOuterUpRight",

    # 眼睛
    "CTRL_expressions_eyeBlinkL": "eyeBlinkLeft",
    "CTRL_expressions_eyeBlinkR": "eyeBlinkRight",
    "CTRL_expressions_eyeWidenL": "eyeWideLeft",
    "CTRL_expressions_eyeWidenR": "eyeWideRight",
    "CTRL_expressions_eyeSquintInnerL": "eyeSquintLeft",
    "CTRL_expressions_eyeSquintInnerR": "eyeSquintRight",
    "CTRL_expressions_eyeCheekRaiseL": "cheekSquintLeft",
    "CTRL_expressions_eyeCheekRaiseR": "cheekSquintRight",
    "CTRL_expressions_eyeFaceScrunchL": "cheekSquintLeft",
    "CTRL_expressions_eyeFaceScrunchR": "cheekSquintRight",
    "CTRL_expressions_eyeLookUpL": "eyeLookUpLeft",
    "CTRL_expressions_eyeLookUpR": "eyeLookUpRight",
    "CTRL_expressions_eyeLookDownL": "eyeLookDownLeft",
    "CTRL_expressions_eyeLookDownR": "eyeLookDownRight",
    "CTRL_expressions_eyeLookLeftL": "eyeLookOutLeft",
    "CTRL_expressions_eyeLookLeftR": "eyeLookInRight",
    "CTRL_expressions_eyeLookRightL": "eyeLookInLeft",
    "CTRL_expressions_eyeLookRightR": "eyeLookOutRight",

    # 鼻子
    "CTRL_expressions_noseWrinkleL": "browInnerUp",  # 没有精确对应，可选择最接近
    "CTRL_expressions_noseWrinkleR": "browInnerUp",
    "CTRL_expressions_noseNostrilDepressL": "noseSneerLeft",
    "CTRL_expressions_noseNostrilDepressR": "noseSneerRight",
    "CTRL_expressions_noseNostrilDilateL": "noseSneerLeft",
    "CTRL_expressions_noseNostrilDilateR": "noseSneerRight",
    "CTRL_expressions_noseNostrilCompressL": "noseSneerLeft",
    "CTRL_expressions_noseNostrilCompressR": "noseSneerRight",
    "CTRL_expressions_noseNasolabialDeepenL": "cheekPuff",
    "CTRL_expressions_noseNasolabialDeepenR": "cheekPuff",

    # 嘴巴
    "CTRL_expressions_mouthCheekBlowL": "cheekPuff",
    "CTRL_expressions_mouthCheekBlowR": "cheekPuff",
    "CTRL_expressions_mouthLeft": "mouthLeft",
    "CTRL_expressions_mouthRight": "mouthRight",
    "CTRL_expressions_mouthUpperLipRaiseL": "mouthUpperUpLeft",
    "CTRL_expressions_mouthUpperLipRaiseR": "mouthUpperUpRight",
    "CTRL_expressions_mouthLowerLipDepressL": "mouthLowerDownLeft",
    "CTRL_expressions_mouthLowerLipDepressR": "mouthLowerDownRight",
    "CTRL_expressions_mouthCornerPullL": "mouthSmileLeft",
    "CTRL_expressions_mouthCornerPullR": "mouthSmileRight",
    "CTRL_expressions_mouthStretchL": "mouthStretchLeft",
    "CTRL_expressions_mouthStretchR": "mouthStretchRight",
    "CTRL_expressions_mouthDimpleL": "mouthDimpleLeft",
    "CTRL_expressions_mouthDimpleR": "mouthDimpleRight",
    "CTRL_expressions_mouthCornerDepressL": "mouthFrownLeft",
    "CTRL_expressions_mouthCornerDepressR": "mouthFrownRight",
    "CTRL_expressions_mouthLipsPurseUL": "mouthPucker",
    "CTRL_expressions_mouthLipsPurseUR": "mouthPucker",
    "CTRL_expressions_mouthLipsPurseDL": "mouthPucker",
    "CTRL_expressions_mouthLipsPurseDR": "mouthPucker",
    "CTRL_expressions_mouthFunnelUL": "mouthFunnel",
    "CTRL_expressions_mouthFunnelUR": "mouthFunnel",
    "CTRL_expressions_mouthFunnelDL": "mouthFunnel",
    "CTRL_expressions_mouthFunnelDR": "mouthFunnel",
    "CTRL_expressions_mouthLipsTogetherUL": "mouthClose",
    "CTRL_expressions_mouthLipsTogetherUR": "mouthClose",
    "CTRL_expressions_mouthLipsTogetherDL": "mouthClose",
    "CTRL_expressions_mouthLipsTogetherDR": "mouthClose",
    "CTRL_expressions_mouthUpperLipBiteL": "mouthRollUpper",
    "CTRL_expressions_mouthUpperLipBiteR": "mouthRollUpper",
    "CTRL_expressions_mouthLowerLipBiteL": "mouthRollLower",
    "CTRL_expressions_mouthLowerLipBiteR": "mouthRollLower",
    "CTRL_expressions_mouthLipsPressL": "mouthPressLeft",
    "CTRL_expressions_mouthLipsPressR": "mouthPressRight",
    "CTRL_expressions_jawOpen": "jawOpen",
    "CTRL_expressions_jawLeft": "jawLeft",
    "CTRL_expressions_jawRight": "jawRight",
    "CTRL_expressions_jawBack": "jawForward",
}

# https://arkit-face-blendshapes.com/
# https://github.com/elijah-atkins/ARKitBlendshapeHelper
arkit_51_dict = {
    "eyeBlinkLeft": 0,
    "eyeLookDownLeft": 1,
    "eyeLookInLeft": 2,
    "eyeLookOutLeft": 3,
    "eyeLookUpLeft": 4,
    "eyeSquintLeft": 5,
    "eyeWideLeft": 6,
    "eyeBlinkRight": 7,
    "eyeLookDownRight": 8,
    "eyeLookInRight": 9,
    "eyeLookOutRight": 10,
    "eyeLookUpRight": 11,
    "eyeSquintRight": 12,
    "eyeWideRight": 13,
    "jawForward": 14,
    "jawLeft": 15,
    "jawRight": 16,
    "jawOpen": 17,
    "mouthClose": 18,
    "mouthFunnel": 19,
    "mouthPucker": 20,
    "mouthLeft": 21,
    "mouthRight": 22,
    "mouthSmileLeft": 23,
    "mouthSmileRight": 24,
    "mouthFrownLeft": 25,
    "mouthFrownRight": 26,
    "mouthDimpleLeft": 27,
    "mouthDimpleRight": 28,
    "mouthStretchLeft": 29,
    "mouthStretchRight": 30,
    "mouthRollLower": 31,
    "mouthRollUpper": 32,
    "mouthShrugLower": 33,
    "mouthShrugUpper": 34,
    "mouthPressLeft": 35,
    "mouthPressRight": 36,
    "mouthLowerDownLeft": 37,
    "mouthLowerDownRight": 38,
    "mouthUpperUpLeft": 39,
    "mouthUpperUpRight": 40,
    "browDownLeft": 41,
    "browDownRight": 42,
    "browInnerUp": 43,
    "browOuterUpLeft": 44,
    "browOuterUpRight": 45,
    "cheekPuff": 46,
    "cheekSquintLeft": 47,
    "cheekSquintRight": 48,
    "noseSneerLeft": 49,
    "noseSneerRight": 50
}


arkit_adjustments = {
    "jawForward": lambda x: -x,                 # 方向反转
    "eyeBlinkLeft": lambda x: x,                # 不变
    "mouthSmileLeft": lambda x: x * 1.2,        # 放大
    "mouthFrownRight": lambda x: x + 0.05,     # 偏移
    # 可以随意扩展更多复杂规则
}


def adjust_arkit_values_lambda(values: np.ndarray, param_names: list):
    """
    values: np.ndarray, shape = (N, 51)
    param_names: list of str, length=51
    """
    adjusted = values.copy()
    
    for i, name in enumerate(param_names):
        if name in arkit_adjustments:
            func = arkit_adjustments[name]
            # 对整列应用 lambda
            adjusted[:, i] = func(adjusted[:, i])
    
    return adjusted

def convert_to_arkit(expressions: np.ndarray, poses: np.ndarray,mat_path: str = "./AboutFace/mat_final.npy") -> np.ndarray:
    """
    将 expressions + jaw pose 转换为 ARKit 51 参数

    Parameters:
    -----------
    expressions : np.ndarray
        表情数据，shape = (N, expr_dim)
    poses : np.ndarray
        pose 数据，shape = (N, pose_dim)
    bone_dict : dict
        骨骼索引字典，至少包含 'jaw'
    mat_path : str
        矩阵路径，用于转换 (51,103)
    
    Returns:
    --------
    y : np.ndarray
        转换后的 ARKit 参数，shape = (N, 51)
    """
    # 1. 加载转换矩阵并计算伪逆
    M = np.load(mat_path)            # (51, 103)
    M_pinv = np.linalg.pinv(M)       # (103, 51) 或者 (51,103) 的伪逆
    
    # 2. 获取 jaw 数据
    jaw_index = bone_dict['jaw']
    jaw_pose = poses[:, jaw_index*3: jaw_index*3+3]  # (N, 3)
    
    # 3. 拼接 expressions + jaw
    x = np.concatenate((expressions, jaw_pose), axis=1)  # (N, 103)
    
    # 4. 转换为 ARKit 参数
    y = x @ M_pinv      # (N, 51)
    
    # 5. 转为 float
    y = y.astype(float)
    
    return y

def format_list(lst, per_line=10, indent=2):
    """把列表格式化成多行字符串"""
    lines = []
    for i in range(0, len(lst), per_line):
        segment = ", ".join(f"{v:.8f}" for v in lst[i:i+per_line])
        lines.append(" " * indent + segment)
    return "[\n" + ",\n".join(lines) + "\n]"


if __name__ == '__main__':
    ''' '''














