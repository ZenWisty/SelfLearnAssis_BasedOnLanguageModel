from langchain.prompts import  ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import List, TypedDict
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from collections import Counter
import random
import os
 
os.environ['DASHSCOPE_API_KEY'] = "<api key>"
llm = ChatTongyi(model='qwen-plus-0624', temperature=0)

# 一个纯基于prompt的推理游戏尝试例子(类似的推理游戏要求底层大模型有长上下文能力)：[./doc/DLReasoningPrompt/Prompt_impl_WolfResident.py](./doc/DLReasoningPrompt/Prompt_impl_WolfResident.py)<br>
# 计划后续利用这个来为思维链提供数据材料
class DataCenterOfPlayer(BaseModel):
    name: str = Field(..., description="the name of the player")
    role: str = Field(..., description="the role of the player")
    other_player: List[str] = Field(..., description="other players")
    live_drug: bool = Field(..., description="if has antidote")
    dead_drug: bool = Field(..., description="if has poison")
    if_alive: bool = Field(..., description="True: alive, False: death")
    info_player_know: str = Field(..., description="other information the player knows")
 
data_center = {
    "Player1": DataCenterOfPlayer(name="Player1", role="", other_player=[], live_drug=False, dead_drug=False, if_alive=True, info_player_know=""),
    "Player2": DataCenterOfPlayer(name="Player2", role="", other_player=[], live_drug=False, dead_drug=False, if_alive=True, info_player_know=""),
    "Player3": DataCenterOfPlayer(name="Player3", role="", other_player=[], live_drug=False, dead_drug=False, if_alive=True, info_player_know=""),
    "Player4": DataCenterOfPlayer(name="Player4", role="", other_player=[], live_drug=False, dead_drug=False, if_alive=True, info_player_know=""),
    "Player5": DataCenterOfPlayer(name="Player5", role="", other_player=[], live_drug=False, dead_drug=False, if_alive=True, info_player_know=""),
    "Player6": DataCenterOfPlayer(name="Player6", role="", other_player=[], live_drug=False, dead_drug=False, if_alive=True, info_player_know=""),
    "Player7": DataCenterOfPlayer(name="Player7", role="", other_player=[], live_drug=False, dead_drug=False, if_alive=True, info_player_know=""),
    "Player8": DataCenterOfPlayer(name="Player8", role="", other_player=[], live_drug=False, dead_drug=False, if_alive=True, info_player_know=""),
    "Player9": DataCenterOfPlayer(name="Player9", role="", other_player=[], live_drug=False, dead_drug=False, if_alive=True, info_player_know="")
}
 
player_names = data_center.keys()
 
def init_role_to_data_center():
    roles = ["狼人"] * 3 + ["平民"] * 3 + ["预言家"] * 1 + ["猎人"] * 1 + ["女巫"] * 1
    random.shuffle(roles)
 
    # 为每个玩家随机分配角色
    for player in data_center.values():
        player.role = roles.pop()
        
 
def init_other_player():
    for player in player_names:
        data_center[player].other_player = [p for p in player_names if p != player]
 
def init_drug_for_nvwu():
    # 初始化女巫角色的药物
    for player in data_center.values():
        if player.role == "女巫":
            player.live_drug = True
            player.dead_drug = True
 
def clear_live_drug_for_nvwu(name):
    data_center[name].live_drug = False
def clear_dead_drug_for_nvwu(name):
    data_center[name].dead_drug = False
        
def set_info_player_know(name, info):
    data_center[name].info_player_know += (info + '\\n')
 
def set_player_status(name: str, if_alive):
    data_center[name].if_alive = if_alive
 
def get_player_status(name: str):
    return data_center[name].if_alive
 
def get_players_by_role(role: str):
    return [player.name for player in data_center.values() if player.role == role]
 
def get_other_player(name):
    return data_center[name].other_player
 
def get_role_by_player(name):
    return data_center[name].role
 
def get_history(name):
    return data_center[name].info_player_know
 
def if_has_live_drug(name):
    return data_center[name].live_drug
 
def if_has_dead_drug(name):
    return data_center[name].dead_drug
 
def get_live_player():
    alive_players = [player.name for player in data_center.values() if player.if_alive]
    return alive_players
 
def werewolves_all_dead():
    for player in data_center.values():
        if player.role == "狼人" and player.if_alive:
            return False
    return True
 
def villagers_all_dead():
    for player in data_center.values():
        if player.role == "平民" and player.if_alive:
            return False
    return True
 
def special_roles_all_dead():
    special_roles = ["预言家", "猎人", "女巫"]
    for player in data_center.values():
        if player.role in special_roles and player.if_alive:
            return False
    return True
 
def print_all_player_role():
    print("玩家实际的角色：")
    for player in data_center.values():
        if_alive = "已挂"
        if player.if_alive:
            if_alive = "活着"
        print(f"{player.name}: {player.role}, {if_alive}")
 
class answer_of_role(BaseModel):
    """the actions of rols"""
 
    analysis: str = Field(description="the analysis of player")
    name: str = Field(description=f'''who you want to kill, and the name must be included in [Player1, Player2,Player3,Player4,Player5,Player6,Player7,Player8,Player9] ''')
 
class answer_of_role_when_vote(BaseModel):
    """the actions of rols"""
 
    analysis: str = Field(description="the analysis of player")
    name: str = Field(description=f'''who you want to vote to die, and the name must be included in[Player1, Player2,Player3,Player4,Player5,Player6,Player7,Player8,Player9]''')
 
class answer_of_role_when_speech(BaseModel):
    """the analysis and speech of rols"""
 
    analysis: str = Field(description="the analysis of player, the content can not be known by other player.")
    speech: str = Field(description="the speech of player, the content is what you want to be known by other player.")
 
class answer_of_predict(BaseModel):
    """answer of prophets"""
 
    analysis: str = Field(description="the analysis of prophets")
    name: str = Field(description=f'''whose identity the prophet wants to know, and the name must be included in [Player1, Player2,Player3,Player4,Player5,Player6,Player7,Player8,Player9]''')
 
class anwser_of_witch_save(BaseModel):
    """the actions of rols"""
 
    analysis: str = Field(description="the analysis of player")
    if_save: bool = Field(description="if the witch save the dead tonight")
 
    
class CurrentState(BaseModel):
    """Description of the current situation of the game"""
    
    round_num: int = Field(description="the round num of the game")
    player_dead: List[str] = Field(description="the players who dead in current turn, if no one set None")
  
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        poems_parts: history sentences
    """
    human_name: str
    current_describe: CurrentState
 
def set_init_state():
    init_role_to_data_center()
    init_other_player()
    init_drug_for_nvwu()
    # print_all_player_role() #临时
    
    while 1:
        your_name = input("请选择一个名字（Player1,Player2,Player3,Player4,Player5,Player6,Player7,Player8,Player9）: ")
        if your_name not in player_names:
            print("游戏中没有这个人参与")
        else:
            break
    human_role = get_role_by_player(your_name)
    
    print(f"你是{your_name}，你的角色是{human_role}")
    current_round = CurrentState(
        round_num = 0,
        player_dead=[]
    )
    
    return {'human_name': your_name, 'current_describe': current_round}
        
 
def chain_with_player_name_construct():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是{player_name}, 是一位出色的“狼人杀”游戏玩家，正在玩“狼人杀”游戏。你是 {role}。
                
            其他玩家包括：
            {other_player}, 
            
            游戏中的所有角色为：
            3个狼人
            3个平民
            1个预言家
            1个猎人
            1个女巫
            
            这是第{round_num}晚.
            
            {history}
            根据你{role}的身份，仔细分析这些信息，不要凭空猜测，最终做出最优决定.
            
            """
            ),
            ("human", "{action}"),
        ]
    )
 
    chain = prompt | llm.with_structured_output(schema=answer_of_role)
    return chain
 
def chain_with_if_witch_save():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是{player_name}, 是一位出色的“狼人杀”游戏玩家，正在玩“狼人杀”游戏，你的角色是{role}。
                
            其他玩家包括：
            {other_player}, 
            
            游戏中的所有角色为：
            3个狼人
            3个平民
            1个预言家
            1个猎人
            1个女巫
            
            这是第{round_num}晚.
            
            {history}
            根据你{role}的身份，仔细分析你所了解的信息，对每个人的身份做出最合理的推理，最终做出最有利于你团队的决定.
            
            """
            ),
            ("human", "{action}"),
        ]
    )
 
    chain = prompt | llm.with_structured_output(schema=anwser_of_witch_save)
    return chain
 
def prophet_chain_construct():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是{player_name}, 是一位出色的“狼人杀”游戏玩家，正在玩“狼人杀”游戏。你是 {role}。
                
            其他玩家包括：
            {other_player}, 
            
            游戏中的所有角色为：
            3个狼人
            3个平民
            1个预言家
            1个猎人
            1个女巫
            
            这是第{round_num}晚.
            
            {history}
            根据你{role}的身份，仔细分析你所了解的信息，对每个人的身份做出最合理的推理，最终做出最有利于你团队的决定.
            
            """
            ),
            ("human", "{action}"),
        ]
    )
 
    chain = prompt | llm.with_structured_output(schema=answer_of_predict)
    return chain
 
def vote_chain_construct():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是{player_name}, 是一位出色的“狼人杀”游戏玩家，正在玩“狼人杀”游戏。你是 {role}。
                
            其他玩家包括：
            {other_player}, 
            
            游戏中的所有角色为：
            3个狼人
            3个平民
            1个预言家
            1个猎人
            1个女巫
            当前是第{round_num}天白天.
            
            {history}
            当前活着的玩家包括：{live_player}。
            根据你{role}的身份，仔细分析你所了解的信息，对每个玩家的身份做出最合理的推理，推理不会被人看到，可以大胆的分析猜测。
            其他玩家的发言可能是假话，注意辨别。
            然后做出最利于你的团队的决定。
            
            """
            ),
            
            ("human", "{action}"),
        ]
    )
 
    chain = prompt | llm.with_structured_output(schema=answer_of_role_when_vote)
    return chain
 
def speech_chain_construct():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是{player_name}, 是一位出色的“狼人杀”游戏玩家，正在玩“狼人杀”游戏。你是 {role}。
                
            其他玩家包括：
            {other_player}, 
            
            游戏中的所有角色为：
            3个狼人
            3个平民
            1个预言家
            1个猎人
            1个女巫
            当前是第{round_num}天白天.
            
            {history}
            当前活着的玩家包括：{live_player}。
            根据你{role}的身份，仔细分析你所了解的信息，对每个玩家的身份做出最合理的推理(analysis)，推理不会被人看到，可以大胆的分析猜测。
            然后做出最利于你的团队的发言(speech)，你的发言会被其他玩家听到，必要时要隐瞒一些情况，甚至可以撒谎。
            注意：很多时候，你需要隐藏你的行为和你的角色。
            其他玩家的发言可能是假话，注意辨别。
            
            给出你的
            推理(analysis): 这些分析应该是玩家内心的推理(analysis)，推理出谁是你的对手，但是不一定会如实说出来
            发言(speech): 你的发言是对其他玩家说的，为了使你的团队获胜，如果你是狼人，应该隐瞒自己的角色，如果你是其他角色，应该视情况决定是不是需要告诉大家。
            你需要分别给出推理(analysis)和发言(speech)。
            
            """
            ),
            
            ("human", "{action}"),
        ]
    )
 
    chain = prompt | llm.with_structured_output(schema=answer_of_role_when_speech)
    return chain
 
prophet_chain = prophet_chain_construct()
common_chain = chain_with_player_name_construct()
witch_chain = chain_with_if_witch_save()
vote_chain = vote_chain_construct()
speech_chain = speech_chain_construct()
 
def scene_1_get_dark(state):
    """
    scene 1
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): the verse info
    """
    
    state['current_describe'].round_num += 1
    state['current_describe'].player_dead.clear()
    print(f"===== 第{state['current_describe'].round_num}天 =====\\n")
    live_players = get_live_player()
    live_palyers_str = ",".join(live_players)
    print(f"玩家剩余: {live_palyers_str}")
    print(f"-- 天黑请闭眼 --")
    info = f"===== 第{state['current_describe'].round_num}夜的信息 =====\\n"
    for player in player_names:
        set_info_player_know(player, info)
 
    return state
 
def get_first_alive_wolf():
    for player in data_center.values():
        if player.role == "狼人" and player.if_alive:
            return player.name
    return None
 
def scene_2_wolf_action(state):
    """
    the action of witch after wolf action
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): the verse info
    """
    print("-- 狼人请睁眼，狼人请杀人 --")
    
    wolf_players = get_players_by_role('狼人')
    random.shuffle(wolf_players)
    
    if state['human_name'] in wolf_players and get_player_status(state['human_name']):
        wolf_live_player = state['human_name']
        print('狼人：', ",".join(wolf_players))
    else:
        wolf_live_player = get_first_alive_wolf()
        
    round_num = state['current_describe'].round_num
    
    
    other_player = get_other_player(wolf_live_player)
    random.shuffle(other_player)
    if round_num == 1:
        for wolf_player in wolf_players:
            wolf_players_copy = wolf_players.copy()
            wolf_players_copy.remove(wolf_player)
            info = f"你知道了你的同伙，也就是其他狼人： {wolf_players_copy[0]} 和 {wolf_players_copy[1]}\\n"
            set_info_player_know(wolf_player, info)
    history = get_history(wolf_live_player)
 
    player_killed = ''
    if state['human_name'] != wolf_live_player:
        try:
            res = common_chain.invoke({'player_name': wolf_live_player, 'role': '狼人', 'other_player': other_player, 'round_num': round_num, 'history':history, 'action': '游戏规定，狼人必须在晚上杀死一个玩家，你要杀谁。'})
        except Exception as e:
            print(f"scene_2_wolf_action: {e}")
            return None
        
        player_killed = res.name.replace('"', '').replace("'", '')
    else:
        live_players = get_live_player()
        
        while(1):
            player_killed = input(f"你作为狼人，要杀死谁({live_players}): ")
            if player_killed in live_players:
                break
 
    state['current_describe'].player_dead.append(player_killed)
    
    info = f"你们，也就是狼人团队杀死了{player_killed}\\n"
    for wolf_player in wolf_players:
        set_info_player_know(wolf_player, info)
    print("-- 狼人请闭眼 --")
 
    return state
 
def scene_2_prediction_action(state):
    """
    the action of prophet after wolf action
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): the verse info
    """
    print("-- 预言家请睁眼 --")
    
    pre_players = get_players_by_role('预言家')[0]
    if not get_player_status(pre_players):
        if state['human_name'] == pre_players:
            print("你是预言家，但已出局")
        return state
    
    round_num = state['current_describe'].round_num
    other_player = get_other_player(pre_players)
    random.shuffle(other_player)
    
    history = get_history(pre_players)
 
    if state['human_name'] != pre_players:
        try:
            res = common_chain.invoke({'player_name': pre_players, 'role': '预言家', 'other_player': other_player, 'round_num': round_num, 'history':history, 'action': '按照游戏规则，你可以知道一名仍然活着的玩家是否是狼人团队，请分析当前形式，然后告诉我你想知道谁的身份。'})
        except Exception as e:
            print(f"scene_2_prediction_action: {e}")
            return None
    
        player = res.name.replace('"', '').replace("'", '')
    else:
        other_players = get_other_player(state['human_name'])
        live_players = get_live_player()
        other_live_player = list(set(other_players) & set(live_players))
        while(1):
            player = input(f"你作为预言家，要查看谁的身份({other_live_player}): ")
            if player in other_live_player:
                break
    
    role_pre = get_role_by_player(player)
    if role_pre == "狼人":
        if_wolf = "是"
    else:
        if_wolf = "不是"
    if state['human_name'] == pre_players:
        print(f"{player}{if_wolf}狼人")
    info = f"你知道了：{player}{if_wolf}狼人\\n"
    set_info_player_know(pre_players, info)
    print("-- 预言家请闭眼 --")
    return state
 
def scene_2_witch_save_action(state):
    """
    the action of witch after dark
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): the verse info
    """
    print("-- 女巫请睁眼，有人被杀，女巫救不救 --")
    witch_players = get_players_by_role('女巫')[0]
    current_dead = state['current_describe'].player_dead[0]
    
    if not get_player_status(witch_players):
        if state['human_name'] == witch_players:
            print("女巫已死")
        set_player_status(current_dead, False)
        return state
    
    round_num = state['current_describe'].round_num
    other_player = get_other_player(witch_players)
    history = get_history(witch_players)
    live_drug_des = ''
    info = f"{current_dead}被狼人弄伤了。"
    if if_has_live_drug(witch_players):
        if state['human_name'] != witch_players:
            live_drug_des = f"{current_dead}被伤了，你有1瓶解药，你要救他吗"
 
            try:
                res = witch_chain.invoke({'player_name': witch_players, 'role': '女巫', 'other_player': other_player, 'round_num': round_num, 'history':history, 'action': live_drug_des})
            except Exception as e:
                print(f"scene_2_witch_save_action: {e}")
                return None
        
            if_to_save = res.if_save
        else:
            while(1):
                YorN = input(f"{info}你是否要救他(Y/N):")
                if YorN.lower() == 'y':
                    if_to_save = True
                    break  # 跳出循环
                elif YorN.lower() == 'n':
                    if_to_save = False
                    break  # 跳出循环
                else:
                    print("请输入 'Y' 或 'N'")
        
        if if_to_save == True:
            set_player_status(current_dead, True)
            state['current_describe'].player_dead.remove(current_dead)
            clear_live_drug_for_nvwu(witch_players)
            info += f"你救了{current_dead}。\\n"
        else:
            set_player_status(current_dead, False)
            info += f"你没有救{current_dead}，他最终死了。\\n"
        set_info_player_know(witch_players, info)
    else:
        set_player_status(current_dead, False)
        info = f"{current_dead}被狼人弄伤了，你没解药再救{current_dead}，他最终死了。\\n"
        set_info_player_know(witch_players, info)
    return state
 
def scene_2_witch_kill_action(state):
    """
    the action of witch after dark
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): the verse info
    """
    print("-- 女巫要杀人吗 --")
    witch_players = get_players_by_role('女巫')[0]
    if not get_player_status(witch_players):
        if state['human_name'] == witch_players:
            print("女巫已死")
        return state
    
    round_num = state['current_describe'].round_num
    other_player = get_other_player(witch_players)
    history = get_history(witch_players)
    live_drug_des = ''
    if if_has_dead_drug(witch_players):
        
        if state['human_name'] != witch_players:
            live_drug_des = f"你有1瓶毒药，你要杀掉一位玩家吗，如果不想就回答None，如果想，就说出他的名字"
 
            try:
                res = common_chain.invoke({'player_name': witch_players, 'role': '女巫', 'other_player': other_player, 'round_num': round_num, 'history':history, 'action': live_drug_des})
            except Exception as e:
                print(f"scene_2_witch_kill_action: {e}")
                return None
    
            player = res.name.replace('"', '').replace("'", '')
        else:
            other_players = get_other_player(state['human_name'])
            live_players = get_live_player()
            other_live_player = list(set(other_players) & set(live_players))
            while(1):
                player = input(f"你作为女巫，要毒杀谁吗({other_live_player})，如果不想，请输入None: ")
                if player in other_live_player or player == 'None':
                    break
        if 'None' == player:
            info = f"第{round_num}轮你没有下毒\\n"
        else:
            info = f"第{round_num}轮，你毒杀了{player}。\\n"
            clear_dead_drug_for_nvwu(witch_players)
            set_player_status(player, False)
            state['current_describe'].player_dead.append(player)
        set_info_player_know(witch_players, info)
    else:
        if state['human_name'] == witch_players:
            print(f"你是女巫，但你已经没有毒药了")
        
    print(f"-- 女巫请闭眼 --")
    return state
 
def scene_3_it_is_dawn(state):
    """
    scene 3
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): the verse info
    """
    print("-- 天亮了，大家请睁眼 --")
    # 初始化一个对象
    player_who_dead = state['current_describe'].player_dead
    round_num = state['current_describe'].round_num
    if len(player_who_dead) == 0:
        player_who_dead = "没有人"
    else:
        player_who_dead = ",".join(player_who_dead)
    print(f"昨晚{player_who_dead}被杀")
    info = f"第{round_num}晚{player_who_dead}被杀\\n\\n"
    info += f"===== 第{round_num}天亮后的信息 =====\\n"
    for player_name in player_names:
        set_info_player_know(player_name, info)
    state['current_describe'].player_dead.clear()
    return state
 
def scene_3_light_speech(state):
    """
    scene 3
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): the verse info
    """
    print("-- 请依次发言 --")
    round_num = state['current_describe'].round_num
    live_player = get_live_player()
    
    action = f'''天亮了，接着大家的对话，发言描述你所掌握的情况.'''
    
    
    info = f"下面是第{round_num}天亮后，没有出局的人展开对话：\\n"
    for player_name in player_names:
        set_info_player_know(player_name, info)
    
    for player_name in player_names:
        if get_player_status(player_name):
            if state['human_name'] != player_name:
                role = get_role_by_player(player_name)
                if role == '预言家':
                    role += ', 发言时注意既能指出你昨晚验证的人的身份，又能隐藏好自己的身份。'
                other_player = get_other_player(player_name)
                random.shuffle(other_player)
                history = get_history(player_name)
                try:
                    res = speech_chain.invoke({'player_name': player_name, 'role': role, 'other_player': other_player, 'round_num': round_num, 'history':history, 'live_player':live_player, 'action': action})
                except Exception as e:
                    print(f"scene_3_light_speech: {e}")
                    continue
                if res == None:
                    print(f'{player_name} 发言: None\\n')
                    continue
                info = f'{player_name}： {res.speech}\\n\\n'
                print(f'{player_name} 发言: {res.speech}\\n')
                
            else:
                while(1):
                    speech = input(f"请{state['human_name']}发言: ")
                    info = f'{player_name}： {speech}\\n'
                    break
                print(f'{player_name} 发言: {speech}')
                
            for player_name in player_names:
                set_info_player_know(player_name, info)
 
    return state
 
def scene_3_light_vote(state):
    """
    scene 3
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): the verse info
    """
    print("-- 请投票 --")
    round_num = state['current_describe'].round_num
    live_player = get_live_player()
    live_player_str = ",".join(live_player)
    
    action = f'''给出你的分析，并选择出你希望出局的对象.'''
    
    info = f"下面是投票结果，显示了每个人都希望谁出局：\\n"
    
    vote = []
    for player_name in player_names:
        if get_player_status(player_name):
            if state['human_name'] != player_name:
                role = get_role_by_player(player_name)
                other_player = get_other_player(player_name)
                random.shuffle(other_player)
                history = get_history(player_name)
                try:
                    res = vote_chain.invoke({'player_name': player_name, 'role': role, 'other_player': other_player, 'round_num': round_num, 'history':history, 'live_player':live_player_str, 'action': action})
                except Exception as e:
                    print(f"scene_3_light_vote: {e}")
                    continue
                info += f'{player_name}投了： {res.name}\\n'
                print(f'{player_name} 投了：{res.name}\\n')
                
                vote.append(res.name)
            else:
                other_players = get_other_player(state['human_name'])
                live_players = get_live_player()
                other_live_player = list(set(other_players) & set(live_players))
                while(1):
                    player = input(f"要投谁({other_live_player}): ")
                    if player in other_live_player:
                        info += f'{player_name}投了： {player}\\n'
                        break
                vote.append(player)
                print(f'{player_name}投了：{player}\\n')
                
    for player_name in player_names:
        set_info_player_know(player_name, info)
 
    # 使用Counter计算每个元素出现的次数
    counter = Counter(vote)
 
    # 找到出现最多的元素
    most_voted_player, voted_time = counter.most_common(1)[0]
 
    set_player_status(most_voted_player, False)
    info = f"第{round_num}轮天亮后，{most_voted_player}被投票投死了.\\n"
    print(f"得票最多的是: {most_voted_player}, 票数是: {voted_time}, 因此他被投死了。")
    
    state['current_describe'].player_dead.append(most_voted_player) 
    
    for player_name in player_names:
        set_info_player_know(player_name, info)
    
    return state
 
def scene_3_light_hunter_kill_after_vote(state):
    """
    scene 3
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): the verse info
    """
 
    most_voted_player = state['current_describe'].player_dead[0]
    rola_of_most_voted_player = get_role_by_player(most_voted_player)
    if rola_of_most_voted_player != '猎人':
        print("被杀的不是猎人")
        return state
 
    print(f"被投死的{most_voted_player}是猎人，因此他有权杀死一名玩家。")
    round_num = state['current_describe'].round_num
    live_players = get_live_player()
    live_players_str = ",".join(live_players)
 
    if state['human_name'] != most_voted_player:
        other_player_of_voted = get_other_player(most_voted_player)
        history_of_voted = get_history(most_voted_player)
        action_of_voted = f"你作为猎人被投票投死了，按照游戏规则，在猎人被投死的时候，可以除掉一名玩家，当前还活着的玩家有{live_players_str}，请分析当前情况，给出你要除掉的对象。如果不确定，也可以不除，给出None。"
        try:
            res_voted = common_chain.invoke({'player_name': most_voted_player, 'role': "猎人", 'other_player': other_player_of_voted, 'round_num': round_num, 'history':history_of_voted, 'action': action_of_voted})
        except Exception as e:
            print(f"scene_3_light_hunter_kill_after_vote: {e}")
            return state
        player_kill_by_hunter = res_voted.name
    else:
        other_players = get_other_player(state['human_name'])
        other_live_player = list(set(other_players) & set(live_players))
        while(1):
            player_kill_by_hunter = input(f"作为猎人，你被投死了，按照规定可以杀掉一个玩家({other_live_player}或None): ")
            if player_kill_by_hunter in other_live_player:
                break
    if 'None' == player_kill_by_hunter:
        info = f"{most_voted_player}的角色是猎人，但临死前他没有杀玩家。\\n"
    else:
        info = f"{most_voted_player}的角色是猎人，按照规则在临死前他可以杀掉一个玩家，于是它把{player_kill_by_hunter}给杀了。\\n"
    print(f"{most_voted_player}(猎人)杀死了{player_kill_by_hunter}")
    set_player_status(player_kill_by_hunter, False)
    
    for player_name in player_names:
        set_info_player_know(player_name, info)
    
    return state
 
def if_play_end(state):
    if special_roles_all_dead() or villagers_all_dead():
        print("===== 游戏结束，狼人获胜 =====")
        print_all_player_role()
        return True
    if werewolves_all_dead():
        print("===== 游戏结束，好人获胜 =====")
        print_all_player_role()
        return True
    
    return False
 
def construct_app():
    workflow = StateGraph(GraphState)
    workflow.add_node("scene_1_get_dark", scene_1_get_dark)
    workflow.add_node("scene_2_wolf_action", scene_2_wolf_action)
    workflow.add_node("scene_2_prediction_action", scene_2_prediction_action)
    workflow.add_node("scene_2_witch_save_action", scene_2_witch_save_action)
    workflow.add_node("scene_2_witch_kill_action", scene_2_witch_kill_action)
    workflow.add_node("scene_3_it_is_dawn", scene_3_it_is_dawn)
    workflow.add_node("scene_3_light_speech", scene_3_light_speech)
    workflow.add_node("scene_3_light_vote", scene_3_light_vote)
    workflow.add_node("scene_3_light_hunter_kill_after_vote", scene_3_light_hunter_kill_after_vote)
    
 
    workflow.set_entry_point("scene_1_get_dark")
    workflow.add_edge("scene_1_get_dark", 'scene_2_wolf_action')
    workflow.add_edge("scene_2_wolf_action", 'scene_2_prediction_action')
    workflow.add_edge("scene_2_prediction_action", 'scene_2_witch_save_action')
    workflow.add_edge("scene_2_witch_save_action", 'scene_2_witch_kill_action')
    workflow.add_edge("scene_2_witch_kill_action", 'scene_3_it_is_dawn')
    workflow.add_conditional_edges(
                                "scene_3_it_is_dawn", 
                                if_play_end, 
                                {
                                    True: END,
                                    False: "scene_3_light_speech",
                                },
                                )
    workflow.add_edge("scene_3_light_speech", 'scene_3_light_vote')
    
    workflow.add_conditional_edges(
                                "scene_3_light_vote", 
                                if_play_end, 
                                {
                                    True: END,
                                    False: "scene_3_light_hunter_kill_after_vote",
                                },
                                )
    
    workflow.add_conditional_edges(
                                "scene_3_light_hunter_kill_after_vote", 
                                if_play_end, 
                                {
                                    True: END,
                                    False: "scene_1_get_dark",
                                },
                                )
    
    app = workflow.compile()
    return app
 
def gen_result():
    RECURSION_LIMIT = 50
    app = construct_app()
    inputs = set_init_state()
    
    for output in app.stream(inputs, {"recursion_limit": RECURSION_LIMIT},):
        for key, value in output.items():
            # Node
            print(f"***")
            # Optional: print full state at each node
            # pprint(value)
    if output == None:
        print("output is none")
        return None
 
gen_result()
 