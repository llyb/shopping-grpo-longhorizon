import gym
import json
import os
import random
import time
import numpy as np

from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import defaultdict
from flask import Flask
from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    get_top_n_product_from_keywords,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    normalize_query,
    ACTION_TO_TEMPLATE,
    PRODUCT_WINDOW,
    SEARCH_RETURN_N,
    END_BUTTON, NEXT_PAGE, PREV_PAGE, BACK_TO_SEARCH,
)
from web_agent_site.engine.goal import get_goals
from web_agent_site.engine.reward import (
    evaluate_abstain,
    evaluate_candidate_eligibility,
    evaluate_purchase,
    fixed_termination,
)
from web_agent_site.engine.termination import EvidenceProgressTracker
from web_agent_site.engine.variant_price import resolve_variant_price
from web_agent_site.engine.observation import (
    build_observation_state,
    page_type_from_name,
)
from web_agent_site.engine.config import (
    ENVIRONMENT_VERSION,
    load_config,
)
from web_agent_site.engine.evidence import (
    candidate_quality,
    new_ledger,
    public_ledger,
    record_observation,
)
from web_agent_site.utils import (BASE_DIR, DEFAULT_FILE_PATH, random_idx)

PROMPT_TEMPLATE_zh="""你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。请注意，商品库中存在大量同类商品，你必须通过合理操作，最终购买到最符合要求的目标商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结果，以及你可执行的操作列表。你需要根据当前状态和可选操作，选择并执行最合适的一步。

操作分为两类：
1. 搜索操作
    格式：search[关键词]
    你可以根据当前需求，自主决定搜索关键词。
    只有在搜索功能可用时，才能使用该操作。
2. 点击操作
    格式：click[值]
    你只能点击当前可用操作列表中的按钮或选项，值必须严格对应操作列表中的内容。

规则说明：
1. 在点击【购买】前，必须至少选择一个商品规格（如颜色、尺码等）。
2. 你的目标是综合所有已知信息，通过搜索、筛选和选择，最终购买到“最符合需求的”商品，而非随意购买任意商品。
3. 如果你给出的操作在当前环境中不可执行，则页面不会有任何变化。
4. 提示：当你点击某一个商品规格时，页面其他信息不会发生变化，但该规格会被选中。请据此判断后续操作。

输出格式
在每一步，你必须按照下面格式输出：
Thought: 简要说明你在当前状态下的思考过程和操作依据。
Action: 用规定格式输出你选择的操作。
"""

PROMPT_TEMPLATE_zh_persona="""你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。
我会提供他们的目标商品（例如“一双鞋”）以及个人文档（例如偏好、预算、使用场景等）请注意，商品库中存在大量同类商品，你必须通过合理操作，，综合分析用户需求和当前页面信息，最终购买到最符合用户要求的商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结果，以及你可执行的操作列表。你需要根据当前状态和可选操作，选择并执行最合适的一步。

操作分为两类：
1. 搜索操作
    格式：search[关键词]
    你可以根据当前需求，自主决定搜索关键词。
    只有在搜索功能可用时，才能使用该操作。
2. 点击操作
    格式：click[值]
    你只能点击当前可用操作列表中的按钮或选项，值必须严格对应操作列表中的内容。

规则说明：
1. 在点击【购买】前，必须至少选择一个商品规格（如颜色、尺码等）。
2. 你的目标是综合所有已知信息，通过搜索、筛选和选择，最终购买到“最符合需求的”商品，而非随意购买任意商品。
3. 如果你给出的操作在当前环境中不可执行，则页面不会有任何变化。
4. 提示：当你点击某一个商品规格时，页面其他信息不会发生变化，但该规格会被选中。请据此判断后续操作。

输出格式
在每一步，你必须按照下面格式输出：
Thought: 简要说明你在当前状态下的思考过程和操作依据。
Action: 用规定格式输出你选择的操作。
"""

app = Flask(__name__)
class WebAgentTextEnv(gym.Env):
    """Gym environment for Text mode of WebShop environment"""
    def __init__(
            self,
            observation_mode='html',
            file_path=DEFAULT_FILE_PATH,
            server=None,
            **kwargs
        ):
        """
        Constructor for text environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        get_image
        filter_goals
        limit_goals
        num_products
        human_goals
        session
        session_prefix
        show_attrs
        """
        super(WebAgentTextEnv, self).__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        self.file_path = file_path

        self.base_url = 'http://127.0.0.1:3000'
        self.server = SimServer(
            self.base_url,
            self.file_path,
            self.kwargs.get('filter_goals'),
            self.kwargs.get('limit_goals', -1),
            self.kwargs.get('num_products'),
            self.kwargs.get('human_goals'),
            self.kwargs.get('show_attrs', False),
            self.kwargs.get('shuffle_goals', False),
            self.kwargs.get('shuffle_num', 20),
            self.kwargs.get('shift_goals', False),
            self.kwargs.get('if_persona', False),
        ) if server is None else server
        self.browser = SimBrowser(self.server)
        self.session = self.kwargs.get('session')
        self.session_prefix = self.kwargs.get('session_prefix')
        if self.kwargs.get('get_image', 0):
            # TODO: self.ids 需要从外部传入或初始化，否则这里会报错
            # self.ids = {url: idx for idx, url in enumerate(self.ids)}
            pass
        self.prev_obs = []
        self.prev_actions = []
        self.num_prev_obs = self.kwargs.get('num_prev_obs', 0)
        self.num_prev_actions = self.kwargs.get('num_prev_actions', 0)
        self.split = self.kwargs.get('split', "test")
        self.if_persona = self.kwargs.get('if_persona', False)
        if self.if_persona:
            self.prompt_template = PROMPT_TEMPLATE_zh_persona
        else:
            self.prompt_template = PROMPT_TEMPLATE_zh
        self.history_init = [
            {'role': 'user', 'content': self.prompt_template},
            {'role': 'assistant', 'content': 'ok'},
        ]
        if self.split == 'test':
            self.init_idx = -1
        else:
            self.init_idx = 194
        self.idx = self.init_idx

    def step(self, action):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        info = None
        self.get_available_actions()

        # Determine action type (click, search) and argument
        try:
            action_name, action_arg = parse_action(action)
        except (ValueError, TypeError, AttributeError):
            # 更具体的异常处理，避免捕获所有异常
            action_name, action_arg = "", ""
        if action_arg is not None:
            action_arg = action_arg.lower()
        if action_name == 'finish':
            status = self.server.finish_without_purchase(self.session)
        elif (action_name == 'search' and
            action_arg is not None and
            action_arg != ''):
            #执行搜索
            status = self.browser.search(action_arg)
        elif (action_name == 'click' and
              action_arg in self.text_to_clickable.keys() and
              action_arg != 'search'):
            status = self.browser.click(action_arg, self.text_to_clickable)
        else:
            status = dict(reward=0, done=False)
            session_state = self.server.user_sessions.get(self.session)
            if isinstance(session_state, dict):
                session_state["invalid_action_count"] = int(
                    session_state.get("invalid_action_count", 0)
                ) + 1

        # Update observation, state with the new action
        ob = self.observation
        session_state = self.server.user_sessions.get(self.session)
        if isinstance(session_state, dict) and not status.get("done"):
            public_observation = self.structured_observation()
            record_observation(
                session_state["evidence_ledger"],
                public_observation,
                step=session_state["progress_tracker"].steps + 1,
            )
            # Candidate eligibility is computed only after the current
            # structured observation has been entered in the actor evidence
            # ledger.  Evaluating from backend product facts in ``item_page``
            # would make an unseen acceptable SKU justify stopping.
            if public_observation.get("page_type") in {"product_detail", "information_subpage"} and session_state.get("asin"):
                product_info = self.server.product_item_dict[session_state["asin"]]
                eligibility = evaluate_candidate_eligibility(
                    product_info,
                    session_state["goal"],
                    evidence=public_ledger(session_state["evidence_ledger"]),
                    selected_options=session_state.get("options") or {},
                    price_resolution=session_state.get("price_resolution"),
                )
                session_state.setdefault("candidate_eligibility", {})[
                    session_state["asin"]
                ] = eligibility
                if eligibility["known_valid"]:
                    session_state.setdefault("known_valid_asins", set()).add(
                        session_state["asin"]
                    )
        if not status.get("done"):
            progress = self.server.record_progress(
                self.session,
                action_name,
                action_arg,
                self.server.visible_asins(self.session, self.browser.current_url),
                current_url=self.browser.current_url,
            )
            status["progress"] = progress
            if progress["termination_reason"]:
                status = self.server.terminate_session(
                    self.session,
                    progress["termination_reason"],
                    progress=progress,
                )
        text_list = [ob]
        self.prev_actions.append(action)
        for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
            if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                text_list.append(self.prev_actions[-i])
            if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                text_list.append(self.prev_obs[-i])
        state = ' [SEP] '.join(text_list[::-1])
        self.prev_obs.append(ob)

        #return state, status['reward'], status['done'], info
        return state, status, info

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        search_bar = html_obj.find(id='search_input')
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_='btn')
        product_links  = html_obj.find_all(class_='product-link')
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {
            f'{b.get_text()}'.lower(): b
            for b in buttons + product_links
        }
        for opt in buying_options:
            opt_value = opt.get('value')
            self.text_to_clickable[f'{opt_value}'] = opt
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )

    def get_image(self):
        """Scrape image from page HTML and return as a list of pixel values"""
        import torch

        html_obj = self._parse_html(self.browser.page_source)
        image_url = html_obj.find(id='product-image')
        if image_url is not None:
            image_url = image_url['src']
            if image_url in self.ids:
                image_idx = self.ids[image_url]
                image = self.feats[image_idx]
                return image
        return torch.zeros(512)

    def get_instruction_text(self):
        """Get corresponding instruction text for current environment session"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text

    def _parse_html(self, html=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj

    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == 'text_rich':
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == 'url':
            return self.state['url']
        else:
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )

    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )

    def convert_html_to_text(self, html, simple=False):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        if simple:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            for t in visible_texts:
                if t == '\n':
                    continue
                if t.parent.name == 'button':  # button
                    processed_t = f'[button] {t} [button_]'
                elif t.parent.name == 'label':  # options
                    if f'"{t}"' in self.state['url']:
                        processed_t = f'  [clicked button] {t} [clicked button_]'
                        observation = f'You have clicked {t}.\n' + observation
                    else:
                        processed_t = f'  [button] {t} [button_]'
                elif t.parent.get('class') == ["product-link"]: # product asins
                    if f'{t}' in self.server.user_sessions[self.session]['asins']:
                        processed_t = f'\n[clicked button] {t} [clicked button_]'
                    else:
                        processed_t = f'\n[button] {t} [button_]'
                else: # regular, unclickable text
                    processed_t =  str(t)
                observation += processed_t + '\n'
            return observation

    def reset(self,idx=None,session=None, instruction_text=None):
        """Create a new session and reset environment variables"""
        session_int = None

        if session is not None:
            self.session = str(session)
            session_int = idx
        elif self.session_prefix is not None:
            self.session = f"{self.session_prefix}-{idx}"
            session_int = idx
        else:
            self.session = idx

        init_url = f'{self.base_url}/{self.session}'
        self.browser.get(init_url, session_id=self.session, session_int=session_int)

        self.text_to_clickable = None
        self.instruction_text = self.get_instruction_text() if instruction_text is None else instruction_text
        obs = self.observation
        # The reset response is the first observation actually shown to the
        # actor.  Store it before any action is processed so a one-step legal
        # purchase can satisfy the Reward v4 evidence gate.  Later action
        # observations are recorded by ``step`` with their progress step.
        session_state = self.server.user_sessions.get(self.session)
        if isinstance(session_state, dict) and session_state.get("evidence_ledger") is not None:
            record_observation(
                session_state["evidence_ledger"],
                self.structured_observation(),
                step=0,
            )
        self.prev_obs = [obs]
        self.prev_actions = []
        self.history = self.history_init[:]
        self.instruction_simple = self.server.goals[idx]['instruction_simple']
        self.goal_options = self.server.goals[idx]['goal_options']
        if self.if_persona:
            user_persona = self.server.goals[idx]['user_persona'].copy()
            if '__reasoning__' in user_persona:
                del user_persona['__reasoning__']
            self.history.extend([
                {'role': 'user', 'content': f"\n用户个人文档：{json.dumps(user_persona, ensure_ascii=False)}"},
                {'role': 'assistant', 'content': 'ok'},
            ])
        if self.server.goals[idx]['user_persona']:
            self.user_persona = self.server.goals[idx]['user_persona'].copy()
            self.reason_key = self.server.goals[idx]['reason_key']
        return obs, None

    def structured_observation(self):
        available_actions = self.get_available_actions()
        page_name = self.server.get_page_name(self.browser.current_url)
        return build_observation_state(
            page_type=page_type_from_name(page_name),
            session=self.server.user_sessions[self.session],
            product_item_dict=self.server.product_item_dict,
            available_actions=available_actions,
        )

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )

class SimServer:
    """Lightweight simulator of WebShop Flask application for generating HTML observations"""
    def __init__(
        self,
        base_url,
        file_path,
        filter_goals=None,
        limit_goals=-1,
        num_products=None,
        human_goals=0,
        show_attrs=False,
        shuffle_goals=False,
        shuffle_num=20,
        shift_goals=False,
        if_persona=False,
    ):
        """
        Constructor for simulated server serving WebShop application

        Arguments:
        filter_goals (`func`) -- Select specific goal(s) for consideration based on criteria of custom function
        limit_goals (`int`) -- Limit to number of goals available
        num_products (`int`) -- Number of products to search across
        human_goals (`bool`) -- If true, load human goals; otherwise, load synthetic goals
        """
        self.environment_version = ENVIRONMENT_VERSION
        config_path = os.environ.get(
            "SHOP_ENV_CONFIG",
            os.path.join(BASE_DIR, "../configs/environment.json"),
        )
        self.environment_config = load_config(config_path)

        # Load all products, goals, and search engine
        self.base_url = base_url
        self.all_products, self.product_item_dict, self.product_prices, _ = \
            load_products(filepath=file_path, num_products=num_products, human_goals=human_goals)
        self.search_engine = init_search_engine(
            num_products=num_products,
            product_filepath=file_path,
        )
        search_config = self.environment_config["search"]
        if int(search_config["top_k"]) != SEARCH_RETURN_N:
            raise ValueError("search top_k differs from the engine runtime")
        if int(search_config["page_size"]) != PRODUCT_WINDOW:
            raise ValueError("search page_size differs from the engine runtime")
        if (
            self.search_engine.manifest.get("field_weights")
            != search_config["field_weights"]
        ):
            raise ValueError("search index weights differ from the config")
        self.existed_goals = True
        self.goals = get_goals(self.all_products, self.product_prices, if_persona=if_persona)
        self.show_attrs = show_attrs
        self.shuffle_goals = shuffle_goals
        self.shuffle_num = shuffle_num
        self.shift_goals = shift_goals
        # Fix outcome for random shuffling of goals
        #random.seed(233)
        #random.shuffle(self.goals)

        # Apply `filter_goals` parameter if exists to select speific goal(s)
        if filter_goals is not None:
            self.goals = [
                goal for (i, goal) in enumerate(self.goals)
                if filter_goals(i, goal)
            ]

        # Imposes `limit` on goals via random selection
        if limit_goals != -1 and limit_goals < len(self.goals):
            self.weights = [goal['weight'] for goal in self.goals]
            self.cum_weights = [0] + np.cumsum(self.weights).tolist()
            idxs = []
            while len(idxs) < limit_goals:
                idx = random_idx(self.cum_weights)
                if idx not in idxs:
                    idxs.append(idx)
            self.goals = [self.goals[i] for i in idxs]
        print(f'Loaded {len(self.goals)} goals.')
        # Set extraneous housekeeping variables
        self.weights = [goal['weight'] for goal in self.goals]
        #累积权重
        self.cum_weights = [0] + np.cumsum(self.weights).tolist()
        self.user_sessions = dict()
        self.search_time = 0
        self.render_time = 0
        self.sample_time = 0
        self.assigned_instruction_text = None  # TODO: very hacky, should remove
        configured_max_steps = int(
            self.environment_config["termination"]["max_steps"]
        )
        self.max_steps = int(os.environ.get("SHOP_MAX_STEPS", configured_max_steps))
        if self.max_steps != configured_max_steps:
            raise ValueError(
                "SHOP_MAX_STEPS differs from the frozen Environment v2 config"
            )

    @app.route('/', methods=['GET', 'POST'])
    def index(self, session_id, **kwargs):
        """Redirect to the search page with the given session ID"""
        html = map_action_to_html(
            'start',
            session_id=session_id,
            instruction_text=kwargs['instruction_text'],
        )
        url = f'{self.base_url}/{session_id}'
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def search_results(self, session_id, **kwargs):
        """Initialize session and return the search results page"""
        session = self.user_sessions[session_id]
        keywords = kwargs['keywords']  # TODO: why is this using kwargs? why not session?
        assert isinstance(keywords, list)
        requested_page = kwargs.get("page")
        page = 1 if requested_page is None else requested_page
        if page < 1:
            raise ValueError("search result page must be positive")
        normalized_query = normalize_query(' '.join(keywords))
        is_new_query = (
            requested_page is None
            or session.get("normalized_query") != normalized_query
            or "search_result_asins" not in session
        )
        session["page"] = page
        session["keywords"] = keywords
        session["asin"] = None
        session["options"] = {}

        if is_new_query:
            session["actions"]["search"] += 1
            session["normalized_query"] = normalized_query
            session.setdefault("distinct_normalized_queries", set()).add(normalized_query)
            old_time = time.time()
            top_n_products = get_top_n_product_from_keywords(
                keywords,
                self.search_engine,
                self.all_products,
                self.product_item_dict,
            )
            self.search_time += time.time() - old_time

            if self.shuffle_goals and self.shuffle_num != 0:
                if len(top_n_products) > self.shuffle_num:
                    first = top_n_products[:self.shuffle_num]
                    random.shuffle(first)
                    top_n_products = first + top_n_products[self.shuffle_num:]
                else:
                    random.shuffle(top_n_products)
            if self.shift_goals and len(top_n_products) > 40:
                top_n_products = (
                    top_n_products[20:40]
                    + top_n_products[:20]
                    + top_n_products[40:]
                )
            session["search_result_asins"] = [product["asin"] for product in top_n_products]
        else:
            top_n_products = [
                self.product_item_dict[asin]
                for asin in session["search_result_asins"]
                if asin in self.product_item_dict
            ]

        total_pages = max(1, (len(top_n_products) + PRODUCT_WINDOW - 1) // PRODUCT_WINDOW)
        if page > total_pages:
            raise ValueError(
                f"search result page {page} is beyond the final page {total_pages}"
            )

        # Get product list from search result asins and get list of corresponding URLs
        products = get_product_per_page(top_n_products, page)
        session["current_page_asins"] = [product["asin"] for product in products]
        session["total_results"] = len(top_n_products)
        session["total_pages"] = total_pages

        keywords_url_string = '+'.join(keywords)
        url = (
            f'{self.base_url}/search_results/{session_id}/'
            f'{keywords_url_string}/{page}'
        )

        # Render HTML search page and record amount of time taken
        old_time = time.time()
        html = map_action_to_html(
            'search',
            session_id=session_id,
            products=products,
            keywords=session["keywords"],
            page=page,
            total=len(top_n_products),
            total_pages=total_pages,
            normalized_query=normalized_query,
            instruction_text=session["goal"]["instruction_text"],
        )
        self.render_time += time.time() - old_time
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_page(self, session_id, **kwargs):
        """Render and return the HTML for a product item page"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        text_to_clickable = kwargs['text_to_clickable']
        clickable = text_to_clickable[clickable_name]

        # Update session logs with information of last product asin selected
        if (clickable.get('class') is not None and
            clickable.get('class')[0] == 'product-link'):
            session["asin"] = clickable_name.upper()
            session["actions"]["asin"] += 1
            session["asins"].add(session["asin"])
        elif clickable.get('name') is not None:
            clickable_key = clickable['name'].lower()
            session["options"][clickable_key] = clickable_name
            session["actions"]["options"] += 1

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        keywords_url_string = '+'.join(session["keywords"])
        option_string = json.dumps(session['options'])

        # 新增：获取当前选中的option和价格
        price_resolution = resolve_variant_price(
            product_info,
            session["options"],
        )
        selected_price = (
            price_resolution["price"]
            if price_resolution["status"] == "pass"
            else None
        )
        session["price_resolution"] = price_resolution
        session["selected_price"] = selected_price
        session["subpage"] = None
        url = (
            f'{self.base_url}/item_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/'
            f'{session["page"]}/{option_string}'
        )

        html = map_action_to_html(
            'click',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
            show_attrs=self.show_attrs,
            selected_price=selected_price,    # 新增
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_sub_page(self, session_id, **kwargs):
        """Render and return the HTML for a product's sub page (i.e. description, features)"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        for k in ACTION_TO_TEMPLATE:
            if clickable_name.lower() == k.lower():
                clickable_name = k
                break

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        session["subpage"] = clickable_name
        session["actions"][clickable_name] += 1
        keywords_url_string = '+'.join(session["keywords"])
        url = (
            f'{self.base_url}/item_sub_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/{session["page"]}/'
            f'{clickable_name}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{clickable_name}]',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def done(self, session_id, **kwargs):
        """Render and return HTML for done page"""
        session = self.user_sessions[session_id]
        goal = self.user_sessions[session_id]['goal']
        purchased_product = self.product_item_dict[session["asin"]]
        session["actions"]["purchase"] += 1
        price = session.get("selected_price")
        if price is None:
            price = self.product_prices.get(session["asin"])

        # Calculate reward for selected product and set variables for page details
        result = evaluate_purchase(
            purchased_product,
            goal,
            selected_options=session["options"],
            price_resolution=session.get("price_resolution"),
            rewards=self.environment_config["reward"],
            evidence=public_ledger(
                session.get("evidence_ledger"),
                redundant_actions=session["progress_tracker"].redundant_actions,
                invalid_actions=session.get("invalid_action_count", 0),
            ),
        )
        reward, info = result.reward, result.to_dict()

        self.user_sessions[session_id]['verbose_info'] = info
        self.user_sessions[session_id]['done'] = True
        self.user_sessions[session_id]['reward'] = reward

        url = (
            f'{self.base_url}/done/{session_id}/'
            f'{session["asin"]}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{END_BUTTON}]',
            session_id=session_id,
            reward=reward,
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        purchase_record = dict(purchased_product)
        purchase_record["price"] = price
        return html, url, reward, info, purchase_record, goal, session["options"]

    def visible_asins(self, session_id, current_url):
        session = self.user_sessions[session_id]
        page_name = self.get_page_name(current_url)
        if page_name == "search_results":
            return tuple(session.get("current_page_asins") or ())
        if page_name in {"item_page", "item_sub_page"} and session.get("asin"):
            return (session["asin"],)
        return ()

    def record_progress(
        self,
        session_id,
        action_name,
        action_arg,
        visible_asins,
        *,
        current_url=None,
    ):
        session = self.user_sessions[session_id]
        tracker = session["progress_tracker"]
        constraint_evidence = []
        asin = session.get("asin")
        eligibility = (
            session.get("candidate_eligibility", {}).get(asin)
            if asin
            else None
        )
        for gate_name, gate in (
            (eligibility or {}).get("hard_gates") or {}
        ).items():
            constraint_evidence.append(
                f"{asin}:{gate_name}:{gate.get('status', 'unknown')}"
            )
        return tracker.record(
            action_name,
            action_arg,
            visible_asins,
            page_type=page_type_from_name(self.get_page_name(current_url or "")),
            selected_options=session.get("options"),
            constraint_evidence=constraint_evidence,
        )

    def finish_without_purchase(self, session_id):
        session = self.user_sessions[session_id]
        tracker = session["progress_tracker"]
        result = evaluate_abstain(
            effective_result_sets=tracker.effective_result_sets,
            opened_candidates=len(session.get("asins") or ()),
            known_acceptable_candidates=len(
                session.get("known_valid_asins") or ()
            ),
            candidate_quality=candidate_quality(
                session.get("evidence_ledger"),
                session.get("goal", {}).get("reward_contract"),
            ),
            waste_cost=min(
                0.05,
                0.01 * session["progress_tracker"].redundant_actions
                + 0.01 * int(session.get("invalid_action_count", 0)),
            ),
            rewards=self.environment_config["reward"],
        )
        return self._terminal_status(session_id, result)

    def terminate_session(self, session_id, reason, *, progress=None):
        status = self._terminal_status(
            session_id,
            fixed_termination(
                reason,
                rewards=self.environment_config["reward"],
            ),
        )
        if progress is not None:
            status["progress"] = progress
        return status

    def _terminal_status(self, session_id, result):
        session = self.user_sessions[session_id]
        info = result.to_dict()
        session["verbose_info"] = info
        session["done"] = True
        session["reward"] = result.reward
        session["termination_reason"] = result.termination_reason
        return {
            "reward": result.reward,
            "reward_detail": info,
            "done": True,
            "purchase": {},
            "goal": {},
            "termination_reason": result.termination_reason,
            "reward_valid": result.reward_valid,
        }

    def receive(self, session_id, current_url, session_int=None, **kwargs):
        """Map action to the corresponding page"""
        status = dict(reward=0.0, done=False)
        with app.app_context(), app.test_request_context():
            # Create/determine goal, instruction_text from current session
            if session_id not in self.user_sessions:
                idx = int(session_int) if session_int is not None else int(session_id)
                goal = self.goals[idx]
                instruction_text = goal['instruction_text']
                self.user_sessions[session_id] = {'goal': goal, 'done': False}
            else:
                instruction_text = \
                    self.user_sessions[session_id]['goal']['instruction_text']
            if self.assigned_instruction_text is not None:
                instruction_text = self.assigned_instruction_text
                self.user_sessions[session_id]['goal']['instruction_text'] = instruction_text
            session = self.user_sessions[session_id]

            if not kwargs:
                # If no action, reset the session variables
                kwargs['instruction_text'] = instruction_text
                html, url = self.index(session_id, **kwargs)
                self.user_sessions[session_id].update(
                    {
                        'keywords': None,
                        'page': None,
                        'asin': None,
                        'asins': set(),
                        'distinct_normalized_queries': set(),
                        'normalized_query': None,
                        'search_result_asins': [],
                        'current_page_asins': [],
                        'total_results': 0,
                        'total_pages': 0,
                        'options': dict(),
                        'selected_price': None,
                        'subpage': None,
                        'price_resolution': None,
                        'candidate_eligibility': {},
                        'known_valid_asins': set(),
                        'evidence_ledger': new_ledger(),
                        'invalid_action_count': 0,
                        'progress_tracker': EvidenceProgressTracker(
                            max_steps=self.max_steps,
                            exact_repeat_limit=int(
                                self.environment_config["termination"][
                                    "exact_repeat_limit"
                                ]
                            ),
                            no_progress_limit=int(
                                self.environment_config["termination"][
                                    "no_progress_limit"
                                ]
                            ),
                            min_new_asins_per_result_set=int(
                                self.environment_config["termination"][
                                    "min_new_asins_per_result_set"
                                ]
                            ),
                            product_open_progress_budget=int(
                                self.environment_config["termination"][
                                    "product_open_progress_budget"
                                ]
                            ),
                            subpage_progress_budget=int(
                                self.environment_config["termination"][
                                    "subpage_progress_budget"
                                ]
                            ),
                            result_set_progress_budget=int(
                                self.environment_config["termination"][
                                    "result_set_progress_budget"
                                ]
                            ),
                        ),
                        'actions': defaultdict(int)
                    }
                )
            elif 'keywords' in kwargs:
                # If search keywords are available, run a search
                html, url = self.search_results(session_id, **kwargs)
            elif 'clickable_name' in kwargs:
                clickable_name = kwargs['clickable_name'].lower()
                if clickable_name == END_BUTTON.lower():
                    # If "buy now" clicked, calculate reward and flag session as terminated
                    html, url, reward, reward_detail, purchased_product, goal, options = self.done(session_id, **kwargs)
                    status['reward'] = reward
                    status['reward_detail'] = reward_detail
                    status['done'] = True
                    status['purchase'] = self.get_purchase_info(purchased_product, options)
                    status['goal'] = goal
                    status['termination_reason'] = reward_detail.get(
                        "termination_reason", "environment_done"
                    )
                    status['reward_valid'] = reward_detail.get("reward_valid", True)
                elif clickable_name == BACK_TO_SEARCH.lower():
                    # Return to the search form without erasing trajectory-level
                    # progress, explored queries, or opened products.
                    html, url = self.index(
                        session_id,
                        instruction_text=instruction_text,
                    )
                    session.update(
                        {
                            "page": None,
                            "asin": None,
                            "current_page_asins": [],
                            "options": {},
                            "selected_price": None,
                            "price_resolution": None,
                            "subpage": None,
                        }
                    )
                elif (clickable_name == NEXT_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "next page" clicked from search results, re-render with `page` enumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] + 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "prev page" clicked from search results, re-render with `page` denumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] - 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_sub_page'):
                    # If "prev page" clicked from sub page, return to corresponding item page
                    html, url = self.item_page(session_id, **kwargs)
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_page'):
                    # If "prev page" clicked from item page, return to search results page
                    html, url = self.search_results(
                        session_id,
                        keywords=session["keywords"],
                        page=session["page"],
                        **kwargs
                    )
                elif clickable_name in [k.lower() for k in ACTION_TO_TEMPLATE]:
                    # Render item_sub_page if clickable is description, features, or reviews
                    html, url = self.item_sub_page(session_id, **kwargs)
                else:
                    # Otherwise, render current item page
                    html, url = self.item_page(session_id, **kwargs)
            return html, url, status

    def get_purchase_info(self, purchase, option):
        purchase_light = {"asin": purchase["asin"],
                          "category": purchase["category"],
                          "query": purchase["query"],
                          "name": purchase["title"],
                          "product_category": purchase["product_category"],
                          "instruction_text": purchase["instruction_text"],
                          "attributes": purchase["Attributes"],
                           "price": purchase["price"],
                           "options":option}
        return purchase_light


    def get_page_name(self, url):
        """Determine which page (i.e. item_page, search_results) the given URL is pointing at"""
        if url is None:
            return None
        page_names = [
            'search_results',
            'item_page',
            'item_sub_page',
            'done'
        ]
        for page_name in page_names:
            if page_name in url:
                return page_name
        return ''  # index page


class SimBrowser:
    """Simulated browser for rendering the HTML source of WebShop environment pages"""
    def __init__(self, server):
        self.server = server
        self.current_url = None
        self.page_source = None
        self.session_id = None

    def get(self, url, session_id=None, session_int=None):
        """Set browser variables to corresponding link, page HTML for URL"""
        self.session_id = url.split('/')[-1] if session_id is None else session_id
        self.page_source, _, _ = \
            self.server.receive(self.session_id, self.current_url, session_int=session_int)
        self.current_url = url

    def click(self, clickable_name, text_to_clickable):
        """Wrapper for `receive` handler for performing click action on current page"""
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                clickable_name=clickable_name,
                text_to_clickable=text_to_clickable,
            )
        return status

    def search(self, keywords):
        """Wrapper for `receive` handler for performing search action on current page"""
        if isinstance(keywords, str):
            keywords = keywords.split(' ')
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                keywords=keywords,
        )
        return status
