Search.setIndex({docnames:["actions/index","bibliography/index","configuration","dynamics/index","dynamics/road/lane","dynamics/road/regulation","dynamics/road/road","dynamics/vehicle/behavior","dynamics/vehicle/controller","dynamics/vehicle/kinematics","environments/highway","environments/index","environments/intersection","environments/merge","environments/parking","environments/roundabout","index","installation","make_your_own","observations/index","quickstart","rewards/index","user_guide"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.index":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["actions/index.rst","bibliography/index.rst","configuration.rst","dynamics/index.rst","dynamics/road/lane.rst","dynamics/road/regulation.rst","dynamics/road/road.rst","dynamics/vehicle/behavior.rst","dynamics/vehicle/controller.rst","dynamics/vehicle/kinematics.rst","environments/highway.rst","environments/index.rst","environments/intersection.rst","environments/merge.rst","environments/parking.rst","environments/roundabout.rst","index.rst","installation.rst","make_your_own.rst","observations/index.rst","quickstart.rst","rewards/index.rst","user_guide.rst"],objects:{"highway_env.envs.common":{"abstract":[18,0,0,"-"],action:[0,0,0,"-"],observation:[19,0,0,"-"]},"highway_env.envs.common.abstract":{AbstractEnv:[18,1,1,""]},"highway_env.envs.common.abstract.AbstractEnv":{PERCEPTION_DISTANCE:[18,2,1,""],__annotations__:[18,2,1,""],__deepcopy__:[18,3,1,""],__init__:[18,3,1,""],__module__:[18,2,1,""],_automatic_rendering:[18,3,1,""],_cost:[18,3,1,""],_is_terminal:[18,3,1,""],_reward:[18,3,1,""],_simulate:[18,3,1,""],action_type:[18,2,1,""],automatic_rendering_callback:[18,2,1,""],call_vehicle_method:[18,3,1,""],change_vehicles:[18,3,1,""],close:[18,3,1,""],configure:[18,3,1,""],default_config:[18,3,1,""],define_spaces:[18,3,1,""],get_available_actions:[18,3,1,""],metadata:[18,2,1,""],observation_type:[18,2,1,""],randomize_behaviour:[18,3,1,""],render:[18,3,1,""],reset:[18,3,1,""],seed:[18,3,1,""],set_preferred_lane:[18,3,1,""],set_route_at_intersection:[18,3,1,""],set_vehicle_field:[18,3,1,""],simplify:[18,3,1,""],step:[18,3,1,""],to_finite_mdp:[18,3,1,""]},"highway_env.envs.common.action":{ActionType:[0,1,1,""],ContinuousAction:[0,1,1,""],DiscreteMetaAction:[0,1,1,""],action_factory:[0,4,1,""]},"highway_env.envs.common.action.ActionType":{__dict__:[0,2,1,""],__module__:[0,2,1,""],__weakref__:[0,2,1,""],act:[0,3,1,""],space:[0,3,1,""],vehicle_class:[0,3,1,""]},"highway_env.envs.common.action.ContinuousAction":{ACCELERATION_RANGE:[0,2,1,""],STEERING_RANGE:[0,2,1,""],__init__:[0,3,1,""],__module__:[0,2,1,""],act:[0,3,1,""],space:[0,3,1,""],vehicle_class:[0,3,1,""]},"highway_env.envs.common.action.DiscreteMetaAction":{ACTIONS_ALL:[0,2,1,""],ACTIONS_LAT:[0,2,1,""],ACTIONS_LONGI:[0,2,1,""],__init__:[0,3,1,""],__module__:[0,2,1,""],act:[0,3,1,""],space:[0,3,1,""],vehicle_class:[0,3,1,""]},"highway_env.envs.common.observation":{AttributesObservation:[19,1,1,""],GrayscaleObservation:[19,1,1,""],KinematicObservation:[19,1,1,""],KinematicsGoalObservation:[19,1,1,""],ObservationType:[19,1,1,""],OccupancyGridObservation:[19,1,1,""],TimeToCollisionObservation:[19,1,1,""],observation_factory:[19,4,1,""]},"highway_env.envs.common.observation.AttributesObservation":{__init__:[19,3,1,""],__module__:[19,2,1,""],observe:[19,3,1,""],space:[19,3,1,""]},"highway_env.envs.common.observation.GrayscaleObservation":{__init__:[19,3,1,""],__module__:[19,2,1,""],_record_to_grayscale:[19,3,1,""],observe:[19,3,1,""],space:[19,3,1,""]},"highway_env.envs.common.observation.KinematicObservation":{FEATURES:[19,2,1,""],__annotations__:[19,2,1,""],__init__:[19,3,1,""],__module__:[19,2,1,""],normalize_obs:[19,3,1,""],observe:[19,3,1,""],space:[19,3,1,""]},"highway_env.envs.common.observation.KinematicsGoalObservation":{__init__:[19,3,1,""],__module__:[19,2,1,""],observe:[19,3,1,""],space:[19,3,1,""]},"highway_env.envs.common.observation.ObservationType":{__dict__:[19,2,1,""],__module__:[19,2,1,""],__weakref__:[19,2,1,""],observe:[19,3,1,""],space:[19,3,1,""]},"highway_env.envs.common.observation.OccupancyGridObservation":{FEATURES:[19,2,1,""],GRID_SIZE:[19,2,1,""],GRID_STEP:[19,2,1,""],__annotations__:[19,2,1,""],__init__:[19,3,1,""],__module__:[19,2,1,""],normalize:[19,3,1,""],observe:[19,3,1,""],space:[19,3,1,""]},"highway_env.envs.common.observation.TimeToCollisionObservation":{__init__:[19,3,1,""],__module__:[19,2,1,""],observe:[19,3,1,""],space:[19,3,1,""]},"highway_env.envs.highway_env":{HighwayEnv:[10,1,1,""]},"highway_env.envs.highway_env.HighwayEnv":{HIGH_SPEED_REWARD:[10,2,1,""],LANE_CHANGE_REWARD:[10,2,1,""],RIGHT_LANE_REWARD:[10,2,1,""],__annotations__:[10,2,1,""],__module__:[10,2,1,""],_cost:[10,3,1,""],_create_road:[10,3,1,""],_create_vehicles:[10,3,1,""],_is_terminal:[10,3,1,""],_reward:[10,3,1,""],default_config:[10,3,1,""],reset:[10,3,1,""],step:[10,3,1,""]},"highway_env.envs.intersection_env":{IntersectionEnv:[12,1,1,""]},"highway_env.envs.intersection_env.IntersectionEnv":{ACTIONS:[12,2,1,""],ACTIONS_INDEXES:[12,2,1,""],ARRIVED_REWARD:[12,2,1,""],COLLISION_REWARD:[12,2,1,""],HIGH_SPEED_REWARD:[12,2,1,""],__annotations__:[12,2,1,""],__module__:[12,2,1,""],_clear_vehicles:[12,3,1,""],_cost:[12,3,1,""],_is_terminal:[12,3,1,""],_make_road:[12,3,1,""],_make_vehicles:[12,3,1,""],_reward:[12,3,1,""],_spawn_vehicle:[12,3,1,""],default_config:[12,3,1,""],has_arrived:[12,3,1,""],reset:[12,3,1,""],step:[12,3,1,""]},"highway_env.envs.merge_env":{MergeEnv:[13,1,1,""]},"highway_env.envs.merge_env.MergeEnv":{COLLISION_REWARD:[13,2,1,""],HIGH_SPEED_REWARD:[13,2,1,""],LANE_CHANGE_REWARD:[13,2,1,""],MERGING_SPEED_REWARD:[13,2,1,""],RIGHT_LANE_REWARD:[13,2,1,""],__annotations__:[13,2,1,""],__module__:[13,2,1,""],_is_terminal:[13,3,1,""],_make_road:[13,3,1,""],_make_vehicles:[13,3,1,""],_reward:[13,3,1,""],default_config:[13,3,1,""],reset:[13,3,1,""]},"highway_env.envs.parking_env":{ParkingEnv:[14,1,1,""]},"highway_env.envs.parking_env.ParkingEnv":{REWARD_WEIGHTS:[14,2,1,""],STEERING_RANGE:[14,2,1,""],SUCCESS_GOAL_REWARD:[14,2,1,""],__annotations__:[14,2,1,""],__module__:[14,2,1,""],_create_road:[14,3,1,""],_create_vehicles:[14,3,1,""],_is_success:[14,3,1,""],_is_terminal:[14,3,1,""],_reward:[14,3,1,""],compute_reward:[14,3,1,""],default_config:[14,3,1,""],reset:[14,3,1,""],step:[14,3,1,""]},"highway_env.envs.roundabout_env":{RoundaboutEnv:[15,1,1,""]},"highway_env.envs.roundabout_env.RoundaboutEnv":{COLLISION_REWARD:[15,2,1,""],HIGH_SPEED_REWARD:[15,2,1,""],LANE_CHANGE_REWARD:[15,2,1,""],RIGHT_LANE_REWARD:[15,2,1,""],__annotations__:[15,2,1,""],__module__:[15,2,1,""],_is_terminal:[15,3,1,""],_make_road:[15,3,1,""],_make_vehicles:[15,3,1,""],_reward:[15,3,1,""],default_config:[15,3,1,""],reset:[15,3,1,""],step:[15,3,1,""]},"highway_env.road":{lane:[4,0,0,"-"],regulation:[5,0,0,"-"],road:[6,0,0,"-"]},"highway_env.road.lane":{AbstractLane:[4,1,1,""],CircularLane:[4,1,1,""],LineType:[4,1,1,""],SineLane:[4,1,1,""],StraightLane:[4,1,1,""]},"highway_env.road.lane.AbstractLane":{DEFAULT_WIDTH:[4,2,1,""],VEHICLE_LENGTH:[4,2,1,""],__annotations__:[4,2,1,""],__dict__:[4,2,1,""],__module__:[4,2,1,""],__weakref__:[4,2,1,""],after_end:[4,3,1,""],distance:[4,3,1,""],heading_at:[4,3,1,""],is_reachable_from:[4,3,1,""],length:[4,2,1,""],line_types:[4,2,1,""],local_coordinates:[4,3,1,""],metaclass__:[4,2,1,""],on_lane:[4,3,1,""],position:[4,3,1,""],width_at:[4,3,1,""]},"highway_env.road.lane.CircularLane":{__init__:[4,3,1,""],__module__:[4,2,1,""],heading_at:[4,3,1,""],line_types:[4,2,1,""],local_coordinates:[4,3,1,""],position:[4,3,1,""],width_at:[4,3,1,""]},"highway_env.road.lane.LineType":{CONTINUOUS:[4,2,1,""],CONTINUOUS_LINE:[4,2,1,""],NONE:[4,2,1,""],STRIPED:[4,2,1,""],__dict__:[4,2,1,""],__module__:[4,2,1,""],__weakref__:[4,2,1,""]},"highway_env.road.lane.SineLane":{__init__:[4,3,1,""],__module__:[4,2,1,""],heading_at:[4,3,1,""],line_types:[4,2,1,""],local_coordinates:[4,3,1,""],position:[4,3,1,""]},"highway_env.road.lane.StraightLane":{__init__:[4,3,1,""],__module__:[4,2,1,""],heading_at:[4,3,1,""],line_types:[4,2,1,""],local_coordinates:[4,3,1,""],position:[4,3,1,""],width_at:[4,3,1,""]},"highway_env.road.regulation":{RegulatedRoad:[5,1,1,""]},"highway_env.road.regulation.RegulatedRoad":{REGULATION_FREQUENCY:[5,2,1,""],YIELDING_COLOR:[5,2,1,""],YIELD_DURATION:[5,2,1,""],__annotations__:[5,2,1,""],__init__:[5,3,1,""],__module__:[5,2,1,""],enforce_road_rules:[5,3,1,""],is_conflict_possible:[5,3,1,""],respect_priorities:[5,3,1,""],step:[5,3,1,""]},"highway_env.road.road":{Road:[6,1,1,""],RoadNetwork:[6,1,1,""]},"highway_env.road.road.Road":{__init__:[6,3,1,""],__module__:[6,2,1,""],__repr__:[6,3,1,""],act:[6,3,1,""],close_vehicles_to:[6,3,1,""],dump:[6,3,1,""],get_log:[6,3,1,""],neighbour_vehicles:[6,3,1,""],step:[6,3,1,""]},"highway_env.road.road.RoadNetwork":{__annotations__:[6,2,1,""],__dict__:[6,2,1,""],__init__:[6,3,1,""],__module__:[6,2,1,""],__weakref__:[6,2,1,""],add_lane:[6,3,1,""],all_side_lanes:[6,3,1,""],bfs_paths:[6,3,1,""],get_closest_lane_index:[6,3,1,""],get_lane:[6,3,1,""],graph:[6,2,1,""],is_connected_road:[6,3,1,""],is_leading_to_road:[6,3,1,""],is_same_road:[6,3,1,""],lanes_list:[6,3,1,""],next_lane:[6,3,1,""],position_heading_along_route:[6,3,1,""],shortest_path:[6,3,1,""],side_lanes:[6,3,1,""],straight_road_network:[6,3,1,""]},"highway_env.vehicle":{behavior:[7,0,0,"-"],controller:[8,0,0,"-"],kinematics:[9,0,0,"-"]},"highway_env.vehicle.behavior":{AggressiveVehicle:[7,1,1,""],DefensiveVehicle:[7,1,1,""],IDMVehicle:[7,1,1,""],LinearVehicle:[7,1,1,""]},"highway_env.vehicle.behavior.AggressiveVehicle":{ACCELERATION_PARAMETERS:[7,2,1,""],LANE_CHANGE_MIN_ACC_GAIN:[7,2,1,""],MERGE_ACC_GAIN:[7,2,1,""],MERGE_TARGET_VEL:[7,2,1,""],MERGE_VEL_RATIO:[7,2,1,""],__module__:[7,2,1,""],target_speed:[7,2,1,""]},"highway_env.vehicle.behavior.DefensiveVehicle":{ACCELERATION_PARAMETERS:[7,2,1,""],LANE_CHANGE_MIN_ACC_GAIN:[7,2,1,""],MERGE_ACC_GAIN:[7,2,1,""],MERGE_TARGET_VEL:[7,2,1,""],MERGE_VEL_RATIO:[7,2,1,""],__module__:[7,2,1,""],target_speed:[7,2,1,""]},"highway_env.vehicle.behavior.IDMVehicle":{ACC_MAX:[7,2,1,""],COMFORT_ACC_MAX:[7,2,1,""],COMFORT_ACC_MIN:[7,2,1,""],DELTA:[7,2,1,""],DISTANCE_WANTED:[7,2,1,""],LANE_CHANGE_DELAY:[7,2,1,""],LANE_CHANGE_MAX_BRAKING_IMPOSED:[7,2,1,""],LANE_CHANGE_MIN_ACC_GAIN:[7,2,1,""],POLITENESS:[7,2,1,""],TIME_WANTED:[7,2,1,""],__init__:[7,3,1,""],__module__:[7,2,1,""],acceleration:[7,3,1,""],act:[7,3,1,""],change_lane_policy:[7,3,1,""],create_from:[7,3,1,""],desired_gap:[7,3,1,""],maximum_speed:[7,3,1,""],mobil:[7,3,1,""],randomize_behavior:[7,3,1,""],recover_from_stop:[7,3,1,""],step:[7,3,1,""],target_speed:[7,2,1,""]},"highway_env.vehicle.behavior.LinearVehicle":{ACCELERATION_PARAMETERS:[7,2,1,""],ACCELERATION_RANGE:[7,2,1,""],STEERING_PARAMETERS:[7,2,1,""],STEERING_RANGE:[7,2,1,""],TIME_WANTED:[7,2,1,""],__init__:[7,3,1,""],__module__:[7,2,1,""],acceleration:[7,3,1,""],acceleration_features:[7,3,1,""],act:[7,3,1,""],add_features:[7,3,1,""],collect_data:[7,3,1,""],lateral_structure:[7,3,1,""],longitudinal_structure:[7,3,1,""],randomize_behavior:[7,3,1,""],steering_control:[7,3,1,""],steering_features:[7,3,1,""],target_speed:[7,2,1,""]},"highway_env.vehicle.controller":{ControlledVehicle:[8,1,1,""],MDPVehicle:[8,1,1,""]},"highway_env.vehicle.controller.ControlledVehicle":{DELTA_SPEED:[8,2,1,""],KP_A:[8,2,1,""],KP_HEADING:[8,2,1,""],KP_LATERAL:[8,2,1,""],MAX_STEERING_ANGLE:[8,2,1,""],PURSUIT_TAU:[8,2,1,""],TAU_A:[8,2,1,""],TAU_DS:[8,2,1,""],__annotations__:[8,2,1,""],__init__:[8,3,1,""],__module__:[8,2,1,""],act:[8,3,1,""],create_from:[8,3,1,""],follow_road:[8,3,1,""],get_routes_at_intersection:[8,3,1,""],plan_route_to:[8,3,1,""],predict_trajectory_constant_speed:[8,3,1,""],set_route_at_intersection:[8,3,1,""],speed_control:[8,3,1,""],steering_control:[8,3,1,""],target_speed:[8,2,1,""]},"highway_env.vehicle.controller.MDPVehicle":{SPEED_COUNT:[8,2,1,""],SPEED_MAX:[8,2,1,""],SPEED_MIN:[8,2,1,""],__annotations__:[8,2,1,""],__init__:[8,3,1,""],__module__:[8,2,1,""],act:[8,3,1,""],index_to_speed:[8,3,1,""],predict_trajectory:[8,3,1,""],speed_to_index:[8,3,1,""]},"highway_env.vehicle.kinematics":{Vehicle:[9,1,1,""]},"highway_env.vehicle.kinematics.Vehicle":{COLLISIONS_ENABLED:[9,2,1,""],DEFAULT_SPEEDS:[9,2,1,""],LENGTH:[9,2,1,""],MAX_SPEED:[9,2,1,""],WIDTH:[9,2,1,""],__init__:[9,3,1,""],__module__:[9,2,1,""],__repr__:[9,3,1,""],__str__:[9,3,1,""],_is_colliding:[9,3,1,""],act:[9,3,1,""],check_collision:[9,3,1,""],clip_actions:[9,3,1,""],create_from:[9,3,1,""],create_random:[9,3,1,""],destination:[9,3,1,""],destination_direction:[9,3,1,""],direction:[9,3,1,""],dump:[9,3,1,""],front_distance_to:[9,3,1,""],get_log:[9,3,1,""],lane_distance_to:[9,3,1,""],make_on_lane:[9,3,1,""],on_road:[9,3,1,""],on_state_update:[9,3,1,""],step:[9,3,1,""],to_dict:[9,3,1,""],velocity:[9,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function"},terms:{"1e6":20,"1st":9,"25m":19,"2e5":20,"abstract":[4,18],"case":[5,6,19],"class":[0,4,5,6,7,8,9,10,12,13,14,15,18,19],"default":[0,7,8,9,18,19],"float":[0,4,5,6,7,8,9,10,12,13,14,15,18,19],"function":[0,4,6,18,19],"import":[0,3,19,20],"int":[0,4,5,6,7,8,9,10,12,13,14,15,18,19,20],"new":[4,5,6,7,8,9,10,14,18],"return":[0,4,5,6,7,8,9,10,12,13,14,15,18],"static":[5,6],"switch":8,"true":[0,4,6,7,9,10,12,13,14,15,18,19],"try":[5,7,16],"while":[0,9,10,13,15,18,20],But:[12,13],For:[0,6,7,19],The:[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,17,18,19,20],Then:0,There:3,These:[9,19],Use:[14,19,20],Using:[8,19],Will:18,__annotations__:[4,5,6,8,10,12,13,14,15,18,19],__deepcopy__:18,__dict__:[0,4,6,19],__doc__:[0,4,6,19],__init__:[0,4,5,6,7,8,9,18,19],__module__:[0,4,5,6,7,8,9,10,12,13,14,15,18,19],__repr__:[6,9],__str__:9,__weakref__:[0,4,6,19],_automatic_rend:18,_clear_vehicl:12,_cost:[10,12,18],_create_road:[10,14],_create_vehicl:[10,14],_from:6,_is_collid:9,_is_success:14,_is_termin:[10,12,13,14,15,18],_make_road:[12,13,15],_make_vehicl:[12,13,15],_record_to_grayscal:19,_reward:[10,12,13,14,15,18],_simul:18,_spawn_vehicl:12,_to:[6,8,18],a_c:7,a_n:7,a_o:7,abbeel:1,abc:4,abcmeta:4,abl:0,about:7,absolut:[6,12,19],abstractenv:[0,18,19],abstractlan:[4,6,9],acc_max:7,acceler:[0,7,8,9,14],acceleration_featur:7,acceleration_paramet:7,acceleration_rang:[0,7],access:[0,12,13,15,18],accident:18,accord:7,accur:[4,6,7,8,9,18,19],achiev:[12,14],achieved_go:14,act:[0,6,7,8,9],action:[2,6,7,8,9,10,12,13,14,15,16,18,20,22],action_dur:8,action_factori:0,action_typ:18,actions_al:0,actions_index:12,actions_lat:0,actions_longi:0,actiontyp:[0,18],actual:[0,6],add:[0,10,14],add_featur:7,add_lan:6,added:12,addit:13,adjac:[10,14],advanc:1,affect:3,after:[2,6,7],after_end:4,agent:[0,10,12,13,14,16,18],aggressivevehicl:7,agre:12,alex:1,alia:4,all:[2,6,7,10,12,14,15,16,18,19],all_side_lan:6,allow:[0,7,8,18],along:[6,8,9,19],also:[6,10,13,19],altch:1,altern:18,altruist:13,alwai:[0,19],among:8,amplitud:4,anaconda:17,andr:1,andrychowicz:1,angl:[0,6,7,8,9],ani:[6,12,13,14,18,19],anoth:[6,9],ansgar:1,anticip:8,api:22,appear:9,appli:[0,8],approach:[13,15],appropri:14,approxim:7,apt:17,arbitr:5,arc:4,arcsin:8,aren:18,arg:18,arn:1,around:[4,19],arrai:[7,14,19],arrived_reward:12,artifici:20,arxiv:1,assign:5,associ:[12,14,15,18],assum:[7,19],attribut:[0,4,5,6,19],attributesobserv:19,author:16,automat:[0,8,15,18],automatic_rendering_callback:18,autonom:[1,16],avail:[0,6,18,20],avoid:[7,10,12,13,15,19],awr:[1,20],axi:19,base:[0,5,7,9,20],baselin:20,batch_siz:20,becaus:7,been:[12,13,15,18],befor:7,behav:[3,5],behavior:[3,5,10,12,13,15,18],behaviour:[5,6,7,10,12,14,15,18],behind:[9,19],being:[7,9],belong:6,below:19,beta:9,between:[4,5,7,8,9,18],beyond:0,bfs_path:6,bibliographi:16,bibtex:16,bicycl:[1,9],bigint:18,bin:19,block:12,bob:1,bolt:16,bool:[0,4,5,6,7,9,10,12,13,14,15,18,19],both:[0,4,7],boundari:18,box:0,brake:[7,9],breadth:6,brigitt:1,budget:18,buffer_s:20,calcul:9,call:[8,10,12,13,14,15,18,19],call_vehicle_method:18,callabl:18,callback:18,can:[0,2,6,7,8,10,12,13,14,15,16,18,19,20],candid:7,captur:18,car:1,cascad:8,cast:9,cell:19,center:[4,7,8,9],centering_posit:[10,12,13,14,15],central:[4,12],chang:[0,1,2,4,7,8,10,15,18,19],change_lane_polici:7,change_vehicl:18,channel:19,check:[9,18],check_collis:9,choos:8,chosen:[7,9],circl:4,circularlan:4,classmethod:[7,8,9,12,13,14,15,18],clip:[0,19],clip_act:9,clockwis:4,close:[7,18],close_vehicles_to:6,closest:[6,8],code:12,coeffici:7,colab:16,collect:[7,16,19],collect_data:7,collid:10,collis:[5,6,7,9,10,12,13,15,22],collision_reward:[10,12,13,15],collisions_en:9,com:[16,17],combin:8,come:[0,12,19],comfort_acc_max:7,comfort_acc_min:7,command:[7,8,9],common:[0,18,19],compar:6,complet:9,compos:[3,6,10,13,14],comput:[1,4,7,8,9,18],compute_reward:14,concaten:6,condit:[14,20],config:[0,2,10,12,13,14,15,18,19],configur:[0,16,18,22],conflict:5,congest:1,connect:6,consid:[6,16,19],consist:[0,1],constant:[8,18,19],constraint:[12,18],contain:[6,9,18,19],continu:[4,14,22],continuous_lin:4,continuousact:[0,14],contribut:[12,16],control:[3,5,7,14,15,18,22],controlledvehicl:[5,7,8],convers:[4,19],convert:[4,8],coordin:[4,6,19],copi:[7,8,9,18],core:[6,9,19],correl:18,correspond:[0,4,6,8,10,14,18,19],cos:9,cos_h:[12,14,19],cost:10,could:12,count:6,cours:12,crash:[9,10,14],creat:[0,7,8,9,10,14,18,20],create_from:[7,8,9],create_random:9,creation:2,credit:14,cruis:[0,8],current:[6,8,9,12,14,15,18,19],current_index:6,curv:4,custom:16,customari:19,customis:[0,19],cut:7,d_0:7,data:[6,7,19],datafram:[6,9,19],ddpg:20,deceler:7,decelr:0,decentr:12,decid:[6,7],decis:[7,10,12,14,15,16,18],deep:18,default_config:[10,12,13,14,15,18],default_spe:9,default_width:4,defensivevehicl:7,defin:[0,2,4,6,10,12,13,14,15,18,19],define_spac:18,definit:0,delai:7,delta:[0,7,8,9],delta_:8,delta_spe:8,demonstr:20,dens:12,densiti:9,depend:[9,17],depth:6,deriv:8,describ:[3,4,5,6,16,19],descript:[3,16,20],desir:[0,7,8,14,19],desired_gap:7,desired_go:14,destin:[6,8,9,12,19],destination_direct:9,detail:16,detect:[9,19],determin:[4,5],determinist:1,dev:17,dict:[0,6,7,8,9,10,12,13,14,15,18,19,20],dictat:7,dictionari:[2,6,19],did:12,differ:[7,19],direct:9,directli:[0,19],dirk:1,disabl:0,discret:[7,8,22],discretemetaact:[0,10,12,13,15],discretis:19,displai:[5,6,19],distanc:[4,6,7,9,18],distance_w:7,distant:18,doe:7,doesn:[6,7,18],doi:1,done:20,dot:[7,8,9],dqn:19,drive:[5,6,9,10,13,16,18,20],driver:7,dump:[6,9],durat:[8,10,12,15],dure:7,dynam:[0,2,5,6,7,8,9,10,12,14,15,16,18,22],each:[0,5,6,8,10,12,19],east:12,edg:[0,6],edouard:16,ego:[0,7,10,12,13,14,15,18],ego_vehicl:7,either:0,eleur:[16,17],els:[6,8],empir:1,enabl:[0,9],enable_lane_chang:7,encod:[6,19],end:[4,6,8,12,14,15,18],end_phas:4,enforc:5,enforce_road_rul:5,enough:7,ensur:[4,18],entiti:[5,6],env:[0,10,12,13,14,15,17,18,19,20],environ:[0,3,10,12,13,14,15,16,19,22],episod:[10,12,13,14,15,20],episode_reward:20,equal:18,equival:0,eras:8,errat:9,evalu:20,even:7,everi:[0,3],exampl:[18,20],except:19,execut:[0,7,10,12,14,15,18],exist:[7,8,9,18],expect:[12,19],experi:[1,20],expon:7,fail:12,fals:[0,4,5,6,10,12,13,14,15,19,20],fast:15,faster:[0,12],feasibl:1,featur:[7,12,14],features_rang:[12,19],fewer:19,ffmpeg:17,field:19,figur:12,filip:1,fill:19,find:[5,6,8],fine:12,finish:6,first:[5,6,18,19],fix:[18,19],flatten:12,florent:1,flow:15,fluid:1,follow:[0,1,6,7,8,15,19],follow_road:8,fong:1,forbidden:4,forward:[8,9],forwardref:4,foster:10,found:6,four:19,frac:[7,8,9],frame:[4,6,9,18,19],fran:1,free:18,frenet:4,frequenc:7,friction:0,from:[4,6,7,8,9,18,20],front:[7,8,9],front_distance_to:9,front_vehicl:7,full:[0,7,10],futur:[8,20],gain:[7,8],gamma:20,gap:7,gather:16,gcc:17,gener:[1,5,6,18],geometri:[4,6],geq:7,get:[4,6,8,16,17,18,19],get_available_act:[0,18],get_closest_lane_index:6,get_lan:6,get_log:[6,9],get_routes_at_intersect:8,git:17,github:[16,17],give:[5,6,19],given:[4,6,7,8,9,10,12,14,15,18],global:4,go_straight:12,goal:[6,14,20],goal_selection_strategi:20,going:[4,6,7],good:12,googl:16,graph:6,graphic:17,graviti:9,grayscal:22,grayscaleobserv:19,grid:22,grid_siz:19,grid_step:19,guid:16,gym:[0,10,12,13,14,15,19,20],handl:15,handler:8,happen:12,hard:12,has:[4,6,12,13,15,17,18,19],has_arriv:12,have:[6,7,14,18],head:[4,6,7,9,14],heading_at:4,helb:1,help:[4,6,7,8,9,18,19],henneck:1,her:20,her_sac_highwai:20,here:20,high:[8,10,13,14],high_speed_reward:[10,12,13,15],higher:7,highwai:[0,11,12,13,15,17,19,20],highway_env:[0,4,5,6,7,8,9,10,12,13,14,15,18,19,20],highwayenv:10,hindsight:[1,20],hm08:[1,20],horizon:[5,19],horizont:12,hot:19,how:[0,3,7,20],howev:5,howpublish:16,hren:1,http:[16,17],human:18,hyperparam:20,idea:14,ideal:12,identifi:6,idl:[0,12,20],idm:7,idmvehicl:[7,10,13,15,18],ieee:1,illustr:19,imag:[18,22],image1:17,implement:[0,4,7,8,10,12,13,14,15,18,20],impos:7,incent:7,includ:[0,5,6],incom:13,incoming_vehicle_destin:15,increas:7,index:[0,6,7,8,9],index_to_spe:8,induc:7,inevit:7,info:[10,12,14,15,18,20],inform:[1,14],infrastructur:6,infti:19,initi:[4,6,7,8,9,10,12,13,14,15,18,19],initial_spac:10,initial_vehicle_count:12,inner:12,input:[6,8,9],instal:16,instanc:[0,6,19],integ:6,integr:9,intellig:[1,7],interdisciplinari:1,intermedi:18,intern:9,intersect:[0,5,6,8,11,20],intersection_env:12,intersectionenv:12,interv:0,invert:8,involv:18,is_conflict_poss:5,is_connected_road:6,is_leading_to_road:6,is_reachable_from:4,is_same_road:6,is_success:20,its:[0,2,4,6,7,8,9,15],itself:17,jam:7,jean:1,jojo:14,jona:1,josh:1,journal:16,junction:13,k_p:8,keep:12,kei:19,kest:1,kinemat:[0,1,3,5,6,7,8,10,12,22],kinematicobserv:19,kinematicsgo:14,kinematicsgoalobserv:19,known:4,kp_a:8,kp_head:8,kp_later:8,kth07:[1,7],kurtosi:14,kwarg:[0,19],lab:6,label:0,landmark:[5,6],lane:[0,1,3,5,6,7,8,9,10,12,13,14,15,18,19],lane_change_delai:7,lane_change_max_braking_impos:7,lane_change_min_acc_gain:7,lane_change_reward:[10,13,15],lane_distance_to:9,lane_index:[6,7,9],lane_index_1:6,lane_index_2:6,lane_left:0,lane_right:0,laneindex:6,lanes_count:10,lanes_list:6,last:[9,10,12,14,15,18,19],lat:[4,8],later:[0,4,6,12],lateral_structur:7,latest:16,layer:[0,20],lead:[6,7],lear:20,learn:20,learning_r:20,lectur:1,left:[6,7,8,12],length:[4,6,9],leurent:16,level:[0,4,8,12],lib:6,libavcodec:17,libavformat:17,libfreetype6:17,libportmidi:17,librari:20,libsdl1:17,libsdl:17,libsmpeg:17,libswscal:17,light:12,like:19,line:[4,8],line_typ:4,linear:[7,8],linearli:10,linearvehicl:7,linetyp:4,list:[0,3,4,5,6,7,8,18,19,20],load:[18,20],local:[4,6,17],local_coordin:4,locat:9,log:[6,9],longi:4,longitudin:[0,4,6,9,12,15],longitudinal_structur:7,look:[6,19],lookahead:8,low:[0,8,13],lower:[7,10,18],mai:[7,19],main:[4,13,18],maintain:[7,13],make:[0,10,12,13,14,15,16,19,22],make_on_lan:9,maneuv:7,manual:[17,22],map:[0,10],mappingproxi:[0,4,6,19],marcin:1,margin:4,martin:1,master:17,match:[6,19],matter:12,max_spe:9,max_steering_angl:8,maxim:[7,18],maximum:[0,7,9,18],maximum_spe:7,mcgrew:1,mdp:18,mdpvehicl:8,meant:18,mechan:0,memo:18,merg:[11,12,15,20],merge_acc_gain:7,merge_env:13,merge_target_vel:7,merge_vel_ratio:7,mergeenv:13,merging_speed_reward:13,meta:22,metaclass__:4,metadata:18,method:[0,2,4,5,7,8,9],metric:[9,18],microscop:1,might:0,minim:[7,18],minimum:[0,7],misc:16,mixer1:17,mlppolici:20,mobil:[1,7],mode:18,model:[1,3,7,8,9,18,20],modifi:[9,18],modul:[0,6,19],monitor:18,more:[7,10,12,13,14,15],most:[0,5,10],move:[3,9],mtrand:[5,6],multilan:10,multipl:18,munir:14,muno:1,must:[0,6,14,17,18],n_sampled_go:20,n_vehicl:12,name:[0,19],natur:19,ndarrai:[0,4,6,7,8,9,10,12,13,14,15,18,19],nearbi:[7,19],need:18,negoti:[12,13],neighbour:[6,9,10],neighbour_vehicl:6,network:[5,6,8,12],neural:1,next:[6,8,10,12,14,15,18],next_lan:6,node:[6,8,12],non:[7,8],none:[0,4,5,6,7,8,9,10,12,13,14,15,18,19],nonetyp:18,norm:14,normal:[14,19],normalactionnois:20,normalize_ob:19,normalize_reward:12,north:12,note:[1,18],notebook:20,notic:7,novel:1,now:[7,13,19],np_random:[5,6],number:[5,6,9,14,18,19],numpi:[0,4,5,6,7,8,9,10,12,13,14,15,17,18,19,20],nut:16,object:[0,4,5,6,9,10,13,18,19],obs:20,observ:[0,1,2,10,12,13,14,15,16,18,22],observation_factori:19,observation_shap:19,observation_typ:18,observationtyp:[18,19],observe_intent:[9,12,19],obstacl:[5,6],obtain:6,occup:22,occupancygrid:19,occupancygridobserv:19,occur:[12,13,15],occurr:[10,12],off:9,offscreen_rend:[10,13,14,15,19],often:[12,18],ois:1,old:7,on_lan:4,on_road:9,on_state_upd:9,one:[7,8,9],ongo:18,onli:[0,7,9,18],opd:20,optim:18,optimist:1,option:[0,4,6,9,16,18,19],order:[0,9,18,19],origin:6,origin_vehicl:9,oscil:4,other:[5,7,8,9,10,12,14,15,18,19],other_vehicles_typ:[10,13,15],out:10,outer:12,output:7,output_lan:7,over:[10,12,13,14,15,19],overal:7,overload:[10,12,13,14,15,18],overrid:7,overridden:9,own:[7,12,16,22],packag:[6,16],page:1,paltchedandrean17:[1,9],panda:[6,9,19],param:[6,10,18,19],paramet:[0,4,5,6,7,8,9,10,12,13,14,15,18,19],parametr:4,parametris:7,park:[11,20],parking_env:14,parkingenv:14,pass:[12,13,15,18,19],path:[6,18],penalti:[13,18],per:20,perception_dist:18,perform:[0,7,8,10,12,13,14,15,18,19],peter:1,phase:4,philip:1,physic:[1,3],pick:6,piec:16,pieter:1,pilot:8,pip:17,placehold:19,plan:[1,6,8,15,20],plan_route_to:8,plasma:1,pleas:16,point:0,polack:1,polici:[7,18,20],policy_frequ:[10,13,14,15,19],policy_kwarg:20,polit:7,popul:[10,12,13,15,18],posit:[4,6,7,9,14],position_devi:12,position_heading_along_rout:6,possibl:[7,15],pre:0,preced:[6,7],precis:12,predict:[5,8,12,19,20],predict_trajectori:8,predict_trajectory_constant_spe:8,preferred_lan:18,prerequisit:16,presenc:12,present:18,preserv:18,principl:20,print:20,prioriti:[4,5,12],problemat:19,process:[0,1,19],project:[6,16,17],propag:9,properti:[0,7,8,9,12],proport:8,provid:[4,8,16,18],proxim:14,pseudorandom:18,psi:[8,9],psi_:8,psi_l:8,psi_r:8,pub:6,publish:16,pulsat:4,purpos:[12,16],pursuit_tau:8,pygam:[17,19],python3:[6,17],python:17,quick:[16,20],quit:12,rachel:1,rad:[0,4,7,8],radiu:4,rai:1,rais:19,ramp:[12,13,15],random:[5,6,9,10,14,18],randomize_behavior:7,randomize_behaviour:18,randomli:[6,9],randomst:[5,6],rang:[0,8,9,19,20],rate:8,rather:0,ratio:9,reach:[7,10,14],reachabl:[4,9],real:20,realist:7,rear_vehicl:7,reason:[7,12],receiv:10,recent:[5,6],recommend:17,record:[1,5,6,18],record_histori:[5,6],recov:7,recover_from_stop:7,rectangl:19,refer:[0,4,6,8,19],regress:7,regul:3,regulatedroad:5,regulation_frequ:5,reinforc:20,rel:[9,19],relat:1,releas:16,remov:18,render:[18,19,20],render_ag:[10,13,14,15],repeat:9,replai:[1,20],repositori:16,repr:[6,9],repres:[6,9,19],reproduc:18,requir:[7,17,19],research:1,reset:[0,10,12,13,14,15,18,19,20],resolut:19,resolv:5,resp:7,respect:[0,7,8],respect_prior:5,respons:9,result:12,revers:7,review:1,reward:[2,10,12,13,14,15,16,18,20,22],reward_weight:14,rgb:19,rgb_arrai:18,right:[4,5,6,7,8,10,12,13,19],right_lane_reward:[10,13,15],rightmost:10,rlss:20,road:[0,4,7,8,9,10,12,13,14,15,18,19,22],road_network:6,road_object:[5,6],roadnetwork:[3,5,6],roadobject:[6,9],room:13,roundabout:[11,20],roundabout_env:15,roundaboutenv:15,rout:[6,7,8,15],row:19,rudimentari:12,rule:5,run:[19,20],sac:20,safe:[7,13],safeti:7,same:[6,7,8,9],same_lan:6,save:[8,20],scale:[10,12,13,14,15,19],scene:19,schedul:12,schneider:1,scienc:1,screen_height:[10,12,13,14,15,19],screen_width:[10,12,13,14,15,19],search:6,second:[5,6],section:3,see:[4,6,7,8,9,18,19],see_behind:[6,19],seed:18,self:[4,6,7,8,9,18,19],separ:0,sequenc:[4,7,8,9],server:19,set:[0,5,6,8,18,19],set_mod:19,set_preferred_lan:18,set_route_at_intersect:[8,18],set_vehicle_field:18,setpoint:0,sever:[0,6,7,10,12,13,14,15,18,19],shape:19,shortest:6,shortest_path:6,should:[5,6,7,18,19],show_trajectori:[10,13,14,15],shuffl:19,side:[4,10],side_lan:6,sign:9,signal:[10,12,18],signatur:[4,6,7,8,9,18,19],similarli:0,simpl:[7,8,12],simpli:12,simplifi:18,simul:[0,1,7,8,10,12,14,15,18,19],simulation_frequ:[10,13,14,15],sin:9,sin_h:[12,14,19],sinc:[12,19],sinelan:4,singl:18,sinusoid:4,site:6,situat:12,size:19,slip:9,slower:[0,12],some:[0,5,6,8,9,10,13,14,18],sometim:12,soon:13,sophist:12,sort:19,sourc:[0,4,5,6,7,8,9,10,12,13,14,15,18,19],south:12,space:[0,9,14,18,19],spawn_prob:12,specif:[10,12,13,14,15,19],specifi:[0,2,8,9],speed:[0,7,8,9,10,13,14,18],speed_control:8,speed_count:8,speed_devi:12,speed_limit:4,speed_max:8,speed_min:8,speed_to_index:8,spot:14,sqrt:7,stabl:20,stable_baselin:20,stack:19,stack_siz:19,stai:[6,10,19],start:[4,6,13,16],start_phas:4,state:[1,7,8,9,10,12,13,14,15,18,19],staticmethod:6,statist:1,steer:[0,7,8,9,14],steering_control:[7,8],steering_featur:7,steering_paramet:7,steering_rang:[0,7,14],step:[5,6,7,9,10,12,14,15,18,19,20],still:18,stop:[5,7,9],store:[6,7,9],str:[6,7,8,9,12,18,19],straight:[4,10,12,13,14,19],straight_road_network:6,straightlan:4,string:6,stripe:4,stuck:7,subclass:0,subvers:17,success:20,success_goal_reward:14,sudo:17,suffer:13,suggest:7,sum:19,supplementari:[4,14],support:7,symposium:1,system:[1,4,9,12],take:[0,5,7,19],tan:9,target:[0,5,6,7,8],target_lane_index:[7,8],target_spe:[7,8],target_veloc:7,task:[10,12,13,14,15,18,20],tau_a:8,tau_d:8,td3:20,term:7,termin:[10,12,14,15,18],text:[7,8],than:[0,7,19],thei:[0,5,7,13,19],them:[5,10,14],thh00:[1,7],thi:[0,4,7,10,12,13,15,17,18,19],thing:12,those:8,though:7,throttl:[0,8],through:[0,3,5],thu:0,tild:7,time:[3,5,7,8,10,22],time_w:7,timer:7,timestep:[5,6,7,8,9,10,12,14,15,18],timetocollis:[13,15,19],timetocollisionobserv:19,titl:16,to_dict:9,to_finite_mdp:18,tobin:1,too:12,top:[0,8],topic:1,topolog:6,track:[0,8],traffic:[1,12,13,15],train:16,trajectori:[1,5,6,8,20],trajectory_timestep:8,transit:[13,20],transport:1,treiber:1,trigger:7,ttf2:17,tupl:[0,4,5,6,7,8,9,10,12,14,15,18],turn:[8,12],tutori:20,two:[3,5,7,8],type:[0,4,5,6,7,8,9,10,12,13,14,15,18,19],unavail:0,under:8,underbrac:7,uniform:19,union:[0,4,7,8,9,10,18],uniqu:6,unsaf:7,until:[5,9,10,12,14,15,18],updat:[8,9,17],url:16,use:[14,16,18],used:[0,7,9,14,18,19,20],useful:18,user:[16,17],using:[0,2,7,17,19,20],usr:6,usual:5,v_0:7,v_r:8,v_y:19,valu:[0,18,19],variat:8,variou:18,vehicl:[0,1,5,6,7,8,9,10,12,13,14,15,18,22],vehicle_class:0,vehicle_class_path:18,vehicle_length:4,vehicles_count:[10,12,19],veloc:[0,5,7,8,9],verbos:20,verg:14,version:17,vertic:12,video:18,viewer:18,wai:[4,5,12],wait:12,want:18,weak:[0,4,6,19],weight:[7,14,19],welcom:12,welind:1,well:[13,15],west:12,what:19,wheel:[7,8,9],when:[5,6,7,8,10,12,13,15,19],where:[0,6,7,8,9,14,18,19],whether:[0,4,5,6,7,18],which:[0,5,6,7,12,14,17,18,19],who:[4,5],whole:18,whose:[6,7],why:7,width:[4,9],width_at:4,within:[6,19],without:[18,19],wojciech:1,wolski:1,won:18,world:[4,6],wrapper:18,written:20,wrong:7,yaw:8,year:16,yield:[5,8],yield_dur:5,yielding_color:5,you:16,your:[16,22],zaremba:1,zero:[10,19]},titles:["Actions","Bibliography","Configuring an environment","Dynamics","Lane","Road regulation","Road","Behavior","Control","Kinematics","Highway","The Environments","Intersection","Merge","Parking","Roundabout","Welcome to highway-env\u2019s documentation!","Installation","Make your own environment","Observations","Getting Started","Rewards","User Guide"],titleterms:{"default":[10,12,13,14,15],"try":20,The:11,action:0,agent:20,all:20,api:[0,4,5,6,7,8,9,10,12,13,14,15,18,19],behavior:7,bibliographi:1,bit:19,cite:16,close:19,colab:20,collis:19,configur:[2,10,12,13,14,15,19],content:16,continu:0,control:[0,8],discret:0,document:16,drive:19,dynam:3,east:19,ego:19,env:16,environ:[2,11,18,20],exampl:19,farther:19,featur:19,get:20,googl:20,grayscal:19,grid:19,guid:22,head:8,highwai:[10,16],how:16,imag:19,instal:17,intersect:12,kinemat:[9,19],lane:4,later:[7,8],latest:17,longitudin:[7,8],make:[18,20],manual:0,merg:13,meta:0,north:19,observ:19,occup:19,one:19,own:18,park:14,posit:8,prerequisit:17,presenc:19,regul:5,releas:17,reward:21,road:[3,5,6],roundabout:15,same:19,slower:19,speed:19,start:20,thi:16,time:19,train:20,ubuntu:17,usag:[10,12,13,14,15],user:22,v_x:19,vehicl:[3,19],welcom:16,window:17,work:16,your:18}})