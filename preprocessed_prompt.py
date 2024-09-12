def high_speed_coupling_prompt():
    prompt = """
    Role: You are an expert in analyzing tenders and in mechanical system design. 
    
    Task:
    Your task is to refer the context given below and list all the points in order to design High Speed Coupling. It is very important to cover all the points present in the context.

    Instructions:
    1. Keep the exact wording of the context wherever possible.
    2. It is of utmost importance to cover all the points. 
    3. Make the important points/words bold
    4. Do not give any additional information that is not present in the context.
    5. Categorize the points under appropriate headings.Headings must be in capital.
     
    Note: Give title as component name in bold and capital and then list all the points. 

    Context:
    Couplings  
    High Speed Coupling  
    For motor rating between 55 kW and 160 kW, high-speed coupling shall be delayed fill type fluid coupling without resilient plate.  
    For motor with rating below 55 kW, spring type resilient flexible coupling shall be used.  
    For motor rating above 160 kW, VVVF drive with spring type resilient coupling with LT squirrel cage induction motors shall be used.  
    Spring type resilient flexible coupling may also be acceptable for motors equipped with VVFDs.  

    All coupling bolts shall be replaceable without shifting the drive components. All couplings shall be provided with sheet metal guards bolted to the base frame. A service factor of 1.5 on motor rating shall be considered for the selection of all couplings. Pin bush couplings shall not be used.  

    Couplings shall be made of forged materials. Rigid couplings shall be used only for connecting intermediate lengths of long shafts rotating at slow speed. For all other cases, flexible/fluid couplings shall be used.  
    Flexible couplings shall preferably be of spring type resilient couplings unless spelt out elsewhere in this specification.  
    Couplings shall be of modern, compact design for given horsepower capacity. Couplings on motor shafts at 100 rpm and over shall be selected with due regard to minimum WR2 for the capacity. All couplings (except fluid couplings) shall have an adequate service factor of minimum 1.5 over motor power. Service factor for fluid couplings shall be as per OEM’s recommendations.  

    Noise level shall be limited to 85 dB at a distance of 1m from the source of noise and at a height of 1.2 m above floor level for all equipment.
"""
    pages = '19,29,37,40,41'
    return prompt,pages



def low_speed_coupling_prompt():
    prompt = """
Role: You are an expert in analyzing tenders and in mechanical system design. 

Task:
Your task is to refer the context given below and list all the points in order to design low Speed Coupling. It is very important to cover all the points present in the context.

Instructions: 
1. Keep the exact wording of the context wherever possible.
2. It is of utmost importance to cover all the points. 
3. Make the important points/words bold
4. Do not give any additional information that is not present in the context.
5. Categorize the points under appropriate headings.Headings must be in capital.
     
Note: Give title as component name in bold and capital and then list all the points. 

Context:
    Low speed coupling between gearbox output shaft and drive pulley shall be full-geared coupling.
 
    All coupling bolts shall be replaceable without shifting the drive components
 
    All couplings shall be provided with sheet metal guards bolted to the base frame. A service factor of 1.5 on motor rating shall be considered for the selection of all couplings. Pin bush couplings shall not be used
 
    Couplings shall be made of forged materials. Rigid couplings shall be used only for connecting intermediate lengths of long shafts rotating at slow speed. For all other cases, flexible/fluid couplings shall be used.
 
    Flexible couplings shall preferably be of spring type resilient couplings unless spelt out elsewhere in this specification
 
    Couplings shall be of modern, compact design for given horse power capacity. Couplings on motor shafts at 100 rpm and over shall be selected with due regard to minimum WR2 for the capacity. All couplings (except fluid couplings) shall have adequate service factor of minimum 1.5 over motor power. Service factor for fluid couplings shall be as per OEM’s recommendations.
 
    Noise level shall be limited to 85 dB at a distance of 1m from the source of noise and at a height of 1.2 m above floor level for all equipment.
"""
    pages = '19,29,37,40,41'
    return prompt,pages


def gear_box_prompt():
    prompt = """
Role: You are an expert in analyzing tenders and in mechanical system design. 

Task:
Your task is to refer the context given below and list all the points in order to design Gear Box. It is very important to cover all the points present in the context.
In case of gear box , information related to shaft is also important to be considered.

Instructions: 
1. Keep the exact wording of the context wherever possible.
2. It is of utmost importance to cover all the points. 
3. Make the important points/words bold
4. Do not give any additional information that is not present in the context.
5. Categorize the points under appropriate headings.Headings must be in capital.
     
Note: Give title as component name in bold and capital and then list all the points. 

Context:

High-speed shafts shall be designed for critical speed. The ratio of critical speed of shaft shall be not less than 1.2.  
ll steel shafting 150 mm or less in diameter and not requiring enlarged portions (as for gear and other hubs) shall be hot rolled and turned, forged or turned cold rolled or cold drawn. All shafting above 150 mm in diameter and requiring enlarged portions shall be forged and machined to size. All forged shafting shall be annealed or normalized before machining and heat-treated, if necessary.  
eflection in line shaft shall not exceed 0.8 mm per metre length. All shafts above 150 mm in diameter shall be ultrasonically tested.  
Gearbox shall be totally enclosed type up to the last stage of reduction. The gearbox housing shall be fabricated/cast steel of minimum 8 mm thickness and shall be stress relieved. Inspection holes with bolted covers shall be provided at appropriate locations. Dip sticks or indicator shall be provided for indicating oil level. Drain plugs shall be provided on all gearboxes. Lifting lugs shall be provided for handling purposes. All gearboxes shall be air-cooled type (without forced cooling).  
Gear transmission must be properly lubricated. In the case of totally enclosed gearboxes, splash system shall be used. All equipment which normally contains lubricant and is dispatched without such lubricant shall have their interior sprayed with a suitable moisture inhibitor to prevent corrosion during transport and storage. Such equipment shall carry clear, legible tagging indicating that it does not contain lubricant.  
The reducers shall be of cut-tooth, hardened & ground parallel shaft splash lubrication type. The mechanical horsepower rating of the reducers shall be not less than 1.5 times the motor nameplate rating/horsepower; the thermal rating of the reducers shall be equal to or better than the motor horsepower for continuous operation under load.  
A suitable service factor shall be applied in selection of reducers. For reducers with electric motor as the prime mover, the following service factors shall be considered on motor power:  
    - Uniform speeds: 1.5  
    - Moderate shocks: 1.75  
    - Heavy shocks: 2.0  
  
Gearbox shall be selected for a mechanical service factor of minimum 1.5 and thermal service factor of minimum 1.0 on motor kW rating.  
All gearboxes shall be generally parallel shaft helical type, wherever possible.  
Overhung or split gears and pinions shall not be used.  
All gears shall be completely enclosed in oil-tight enclosure.  
All gear shafts shall be supported in anti-friction bearings mounted in the gearbox.  
Splash lubrication system shall be used.  
The housing for gearboxes shall be of cast steel or fabricated. Fabricated gearboxes shall be stress relieved.  
Covers shall be split horizontally at each shaft centerline and fastened and arranged so that the top half can be removed for inspection and repair without disturbing the bottom half.  
The gearboxes shall be provided with breather vents, oil level indicators, and easily accessible drain plugs. Permanent magnet plugs shall also be provided in the gearbox.  
Gearbox shall have a machined base.  
Oil seal arrangement shall be of special design to suit dusty surroundings.  

All inclined conveyors shall be provided with suitable holdback devices to prevent belt running back in case of conveyor stoppage. Holdback ratings shall be minimum 1.5 times the maximum calculated torque.  

Noise level shall be limited to 85 dB at a distance of 1m from the source of noise and at a height of 1.2 m above floor level for all equipment.
"""
    pages = '18,19,29,39,40,42'
    return prompt ,pages

