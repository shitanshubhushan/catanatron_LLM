from catanatron import Player
from catanatron_experimental.cli.cli_players import register_player
from openai import OpenAI
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@register_player("GEMINI2_0")
class Gemini2_0Player(Player):
    def __init__(self, color, is_bot=True):
        super().__init__(color, is_bot)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.max_retries = 2
        self.retry_delay = 1  # seconds

    def decide(self, game, playable_actions):
        """Uses LLM to choose best action using structured output.
        
        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        state_output = self.get_state_string(game, playable_actions)
        
        # Add debug prints
        player_idx = game.state.color_to_index[self.color]
        key = f"P{player_idx}"
        #print(f"\nGemini 2.0 Turn {game.state.num_turns}")
        #print(f"Current VP: {game.state.player_state[f'{key}_VICTORY_POINTS']}")
        
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    extra_body={},
                    model="google/gemini-2.0-flash-001",
                    messages=[{
                        "role": "user",
                        "content": f"""
                        You are an AI that plays Catan. Given a game state, you should analyze it and choose the best possible move from the list of actions. Only return the action number.
                        {state_output}
                        Only return the action number
                        """
                    }],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "catan_move",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "selected_action": {
                                        "type": "integer",
                                        "description": "The number of the selected action from the available actions list",
                                        "minimum": 0,
                                        "maximum": len(playable_actions) - 1
                                    }
                                },
                                "required": ["selected_action"],
                                "additionalProperties": False
                            }
                        }
                    }
                )
                
                #print(f"LLM Output: {completion.choices[0].message.content}")  # Print LLM output
                
                if not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return playable_actions[0]
                
                try:
                    # Try parsing as JSON first
                    content = json.loads(completion.choices[0].message.content)
                    action_idx = content["selected_action"]
                    
                    # If we get here, we have a valid action_idx
                    if 0 <= action_idx < len(playable_actions):
                        return playable_actions[action_idx]  # Return immediately on success
                    
                except (json.JSONDecodeError, KeyError):
                    # If JSON parsing fails, try parsing as plain integer
                    try:
                        action_idx = int(completion.choices[0].message.content.strip())
                        if 0 <= action_idx < len(playable_actions):
                            return playable_actions[action_idx]  # Return immediately on success
                    except ValueError:
                        pass  # Let it continue to next attempt
                
                # If we get here, the action was invalid
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                print(f"State that caused the error:\n{state_output}")
                if 'completion' in locals():
                    print(f"Model response:\n{completion}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                
            # If we get here, all attempts failed or were invalid
            return playable_actions[0]

    def get_state_string(self, game, playable_actions):
        """Returns the game state as a formatted string."""
        output = []
        output.append("=== DEBUG BOT STATE ===")
        # Game Phase Information
        output.append("Game Phase:")
        output.append(f"  Current Prompt: {game.state.current_prompt}")
        output.append(f"  Initial Build Phase: {game.state.is_initial_build_phase}")
        output.append(f"  Turn Number: {game.state.num_turns}")
        
        # Player Information
        output.append("\nPlayers:")
        output.append(f"I am playing as: {self.color}")
        for color in game.state.colors:
            player_idx = game.state.color_to_index[color]
            key = f"P{player_idx}"
            output.append(f"\n{color}{'(ME)' if color == self.color else ''}:")
            output.append(f"  Victory Points: {game.state.player_state[f'{key}_VICTORY_POINTS']}")
            if color == self.color:  # Only show detailed resources for self
                output.append(f"  Resources:")
                output.append(f"    Wood: {game.state.player_state[f'{key}_WOOD_IN_HAND']}")
                output.append(f"    Brick: {game.state.player_state[f'{key}_BRICK_IN_HAND']}")
                output.append(f"    Sheep: {game.state.player_state[f'{key}_SHEEP_IN_HAND']}")
                output.append(f"    Wheat: {game.state.player_state[f'{key}_WHEAT_IN_HAND']}")
                output.append(f"    Ore: {game.state.player_state[f'{key}_ORE_IN_HAND']}")
                output.append(f"  Development Cards:")
                output.append(f"    Knights: {game.state.player_state[f'{key}_KNIGHT_IN_HAND']}")
                output.append(f"    Victory Points: {game.state.player_state[f'{key}_VICTORY_POINT_IN_HAND']}")
                output.append(f"    Year of Plenty: {game.state.player_state[f'{key}_YEAR_OF_PLENTY_IN_HAND']}")
                output.append(f"    Monopoly: {game.state.player_state[f'{key}_MONOPOLY_IN_HAND']}")
                output.append(f"    Road Building: {game.state.player_state[f'{key}_ROAD_BUILDING_IN_HAND']}")
            else:  # Just show total cards for others
                total_resources = (
                    game.state.player_state[f'{key}_WOOD_IN_HAND'] +
                    game.state.player_state[f'{key}_BRICK_IN_HAND'] +
                    game.state.player_state[f'{key}_SHEEP_IN_HAND'] +
                    game.state.player_state[f'{key}_WHEAT_IN_HAND'] +
                    game.state.player_state[f'{key}_ORE_IN_HAND']
                )
                output.append(f"  Total Resource Cards: {total_resources}")
        
        # Board State
        output.append("\nBoard State:")
        output.append(f"  Buildings: {game.state.buildings_by_color}")
        output.append(f"  Resource Bank: {game.state.resource_freqdeck}")
        output.append(f"  Development Cards Left: {len(game.state.development_listdeck)}")
        
        # Special States
        if game.state.is_discarding:
            output.append("\nDiscard Phase Active")
        if game.state.is_moving_knight:
            output.append("\nKnight Movement Active")
        if game.state.is_resolving_trade:
            output.append(f"\nTrade Active: {game.state.current_trade}")
        
        output.append("\nPlayable Actions:")
        for i, action in enumerate(playable_actions):
            output.append(f"{i}: {action}")
        output.append("=====================")
        
        return "\n".join(output) 
    
@register_player("OPENAI_GPT4O_MINI")
class OpenAIGPT4OMiniPlayer(Player):
    def __init__(self, color, is_bot=True):
        super().__init__(color, is_bot)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.max_retries = 2
        self.retry_delay = 1  # seconds

    def decide(self, game, playable_actions):
        state_output = self.get_state_string(game, playable_actions)
        
        # Add debug prints
        player_idx = game.state.color_to_index[self.color]
        key = f"P{player_idx}"
        #print(f"\nGPT4O Mini Turn {game.state.num_turns}")
        #print(f"Current VP: {game.state.player_state[f'{key}_VICTORY_POINTS']}")
        
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    extra_body={
                    },
                    model="openai/gpt-4o-mini",
                    messages=[{
                        "role": "system",
                        "content": """You are a Catan-playing AI that returns moves in JSON format."""
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze this game state and choose the best possible move from the list of actions.
                        Only return a number between 0 and {len(playable_actions) - 1}.
                        {state_output}"""
                    }],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "catan_move",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "selected_action": {
                                        "type": "integer",
                                        "description": "The number of the selected action from the available actions list"
                                    }
                                },
                                "required": ["selected_action"],
                                "additionalProperties": False
                            }
                        }
                    }
                )
                
                #print(f"LLM Output: {completion.choices[0].message.content}")  # Print LLM output
                
                if not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return playable_actions[0]
                
                try:
                    # Try parsing as JSON first
                    content = json.loads(completion.choices[0].message.content)
                    action_idx = content["selected_action"]
                    
                    # If we get here, we have a valid action_idx
                    if 0 <= action_idx < len(playable_actions):
                        return playable_actions[action_idx]  # Return immediately on success
                    
                except (json.JSONDecodeError, KeyError):
                    # If JSON parsing fails, try parsing as plain integer
                    try:
                        action_idx = int(completion.choices[0].message.content.strip())
                        if 0 <= action_idx < len(playable_actions):
                            return playable_actions[action_idx]  # Return immediately on success
                    except ValueError:
                        pass  # Let it continue to next attempt
                
                # If we get here, the action was invalid
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                print(f"State that caused the error:\n{state_output}")
                if 'completion' in locals():
                    print(f"Model response:\n{completion}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                
            # If we get here, all attempts failed or were invalid
            return playable_actions[0]

    def get_state_string(self, game, playable_actions):
        """Returns the game state as a formatted string."""
        output = []
        output.append("=== DEBUG BOT STATE ===")
        # Game Phase Information
        output.append("Game Phase:")
        output.append(f"  Current Prompt: {game.state.current_prompt}")
        output.append(f"  Initial Build Phase: {game.state.is_initial_build_phase}")
        output.append(f"  Turn Number: {game.state.num_turns}")
        
        # Player Information
        output.append("\nPlayers:")
        output.append(f"I am playing as: {self.color}")
        for color in game.state.colors:
            player_idx = game.state.color_to_index[color]
            key = f"P{player_idx}"
            output.append(f"\n{color}{'(ME)' if color == self.color else ''}:")
            output.append(f"  Victory Points: {game.state.player_state[f'{key}_VICTORY_POINTS']}")
            if color == self.color:  # Only show detailed resources for self
                output.append(f"  Resources:")
                output.append(f"    Wood: {game.state.player_state[f'{key}_WOOD_IN_HAND']}")
                output.append(f"    Brick: {game.state.player_state[f'{key}_BRICK_IN_HAND']}")
                output.append(f"    Sheep: {game.state.player_state[f'{key}_SHEEP_IN_HAND']}")
                output.append(f"    Wheat: {game.state.player_state[f'{key}_WHEAT_IN_HAND']}")
                output.append(f"    Ore: {game.state.player_state[f'{key}_ORE_IN_HAND']}")
                output.append(f"  Development Cards:")
                output.append(f"    Knights: {game.state.player_state[f'{key}_KNIGHT_IN_HAND']}")
                output.append(f"    Victory Points: {game.state.player_state[f'{key}_VICTORY_POINT_IN_HAND']}")
                output.append(f"    Year of Plenty: {game.state.player_state[f'{key}_YEAR_OF_PLENTY_IN_HAND']}")
                output.append(f"    Monopoly: {game.state.player_state[f'{key}_MONOPOLY_IN_HAND']}")
                output.append(f"    Road Building: {game.state.player_state[f'{key}_ROAD_BUILDING_IN_HAND']}")
            else:  # Just show total cards for others
                total_resources = (
                    game.state.player_state[f'{key}_WOOD_IN_HAND'] +
                    game.state.player_state[f'{key}_BRICK_IN_HAND'] +
                    game.state.player_state[f'{key}_SHEEP_IN_HAND'] +
                    game.state.player_state[f'{key}_WHEAT_IN_HAND'] +
                    game.state.player_state[f'{key}_ORE_IN_HAND']
                )
                output.append(f"  Total Resource Cards: {total_resources}")
        
        # Board State
        output.append("\nBoard State:")
        output.append(f"  Buildings: {game.state.buildings_by_color}")
        output.append(f"  Resource Bank: {game.state.resource_freqdeck}")
        output.append(f"  Development Cards Left: {len(game.state.development_listdeck)}")
        
        # Special States
        if game.state.is_discarding:
            output.append("\nDiscard Phase Active")
        if game.state.is_moving_knight:
            output.append("\nKnight Movement Active")
        if game.state.is_resolving_trade:
            output.append(f"\nTrade Active: {game.state.current_trade}")
        
        output.append("\nPlayable Actions:")
        for i, action in enumerate(playable_actions):
            output.append(f"{i}: {action}")
        output.append("=====================")
        
        return "\n".join(output) 