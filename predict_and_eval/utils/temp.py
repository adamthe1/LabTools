self.wait = [
            # Frailty - grip strength and leg muscle mass directly affect gait
            {"frailty_gait": ["hand_grip_left", "hand_grip_right", 
                             "body_comp_leg_right_lean_mass", "body_comp_leg_left_lean_mass"]},
            
            # Cardiovascular - orthostatic response, heart rate, hypertension, ECG intervals
            {"cardiovascular_gait": ["standing_one_min_blood_pressure_systolic", 
                                     "standing_three_min_blood_pressure_diastolic",
                                     "from_r_thigh_to_r_ankle_pwv",
                                     "hr_bpm", "Hypertension", "qr_ms"]},
            
            # Sleep - affects daytime fatigue and cognitive function
            {"sleep_gait": ["ahi", "percent_of_wake_time"]},
            
            # Mental - fatigue, mood, and depression affect gait patterns
            {"mental_gait": ["tired_or_little_energy_fortnight", "happiness_level", "Depression"]},
            
            # Medical conditions - directly affect gait mechanics
            {"medical_gait": ["Back Pain", "Fibromyalgia", "Diabetes", "B12 Deficiency"]},
            
            # Age/Gender/BMI - classic confounders, good baseline comparison
            {"demographics_gait": ["age", "bmi"]},
            
            # Lifestyle - physical activity levels
            {"lifestyle_gait": ["walking_minutes_day", "vigorous_activity_minutes"]},
            
            # Exercise wearable - monthly activity tracking
            {"exercise_gait": ["wearable_walking_monthly_hours", "wearable_running_monthly_hours",
                               "wearable_weightlifting_monthly_hours",
                              "wearable_total_monthly_hours"]},
            
            # Nightingale metabolomics - inflammation, muscle metabolism, energy
            {"nightingale_gait": ["GlycA", "Lactate", "Glucose", "Total_BCAA", 
                                  "Creatinine", "DHA", "Omega_3", "Albumin"]}
        ]