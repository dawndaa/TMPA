DATASET_CONFIGS = {
    "openearthmap": {
        "448": {
            "prob_thd": 0.00,
            "cls_token_lambda": -0.35,
            "logit_scale": 50,
            "logit_weight": 1.0,
            "bg_idx": 0,
            "alpha": 0.02,
        }
    },

    "loveda": {
        "448": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.30,
            "logit_scale": 40,
            "logit_weight": 1.0,
            "bg_idx": 0,
            "alpha": 0.02,
        }
    },
    
    "isaid": {
        "448": {
            "prob_thd": 0.20,
            "cls_token_lambda": -0.0,
            "logit_scale": 50,
            "logit_weight": 1.0,
            "bg_idx": 0,
            "alpha": 0.0,
        }
    },
    
    "potsdam": {
        "448": {
            "prob_thd": 0.0,
            "cls_token_lambda": -0.1,
            "logit_scale": 50,
            "logit_weight": 0.0,
            "bg_idx": 5,
            "alpha": 0.02,
        }
    },
    
    "vaihingen": {
        "448": {
            "prob_thd": 0.0,
            "cls_token_lambda": -0.48,
            "logit_scale": 48,
            "logit_weight": 1.0,
            "bg_idx": 5,
            "alpha": 0.02,
        }
    },  
    
    "uavid": {
        "448": {
            "prob_thd": 0.05,
            "cls_token_lambda": -0.3,
            "logit_scale": 45,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.02,
        }
    },  
    
    "udd5": {
        "448": {
            "prob_thd": 0.0,
            "cls_token_lambda": -0.2,
            "logit_scale": 20,
            "logit_weight": 0.0,
            "bg_idx": 4,
            "alpha": 0.0,
        }
    }, 
    
    "vdd": {
        "448": {
            "prob_thd": 0.0,
            "cls_token_lambda": -0.3,
            "logit_scale": 45,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        }
    }, 
    
    "whu_aerial": {
        "448": {
            "prob_thd": 0.19,
            "cls_token_lambda": -0.0,
            "logit_scale": 35,
            "logit_weight": 1.0,
            "bg_idx": 0,
            "alpha": 0.02,
        },
        "896": {
            "prob_thd": 0.19,
            "cls_token_lambda": -0.0,
            "logit_scale": 35,
            "logit_weight": 1.0,
            "bg_idx": 0,
            "alpha": 0.02,
        }
    }, 
    
    "whu_sat": {
        "448": {
            "prob_thd": 0.30,
            "cls_token_lambda": -0.45,
            "logit_scale": 50,
            "logit_weight": 1.0,
            "bg_idx": 0,
            "alpha": 0.0,
        }
    }, 
    
    "inria": {
        "448": {
            "prob_thd": 0.20,
            "cls_token_lambda": -0.10,
            "logit_scale": 35,
            "logit_weight": 0.8,
            "bg_idx": 0,
            "alpha": 0.02,
        },
        "896": {
            "prob_thd": 0.15,
            "cls_token_lambda": -0.10,
            "logit_scale": 35,
            "logit_weight": 0.8,
            "bg_idx": 0,
            "alpha": 0.02,
        }
    },

    "xbd": {
        "448": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.30,
            "logit_scale": 25,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        },
        "896": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.30,
            "logit_scale": 25,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        }
    },
    
    "chn6-cug": {
        "448": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.50,
            "logit_scale": 40,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        },
        "896": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.50,
            "logit_scale": 40,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        }
    },
    
    "deepglobe": {
        "448": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.40,
            "logit_scale": 40,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        },
        "896": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.40,
            "logit_scale": 40,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        }
    },
    
    "massachusetts": {
        "448": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.35,
            "logit_scale": 20,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        },
        "896": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.35,
            "logit_scale": 20,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        }
    },
    
    "spacenet": {
        "448": {
            "prob_thd": 0.39,
            "cls_token_lambda": -0.3,
            "logit_scale": 50,
            "logit_weight": 1.0,
            "bg_idx": 0,
            "alpha": 0.02,
        },
        "896": {
            "prob_thd": 0.42,
            "cls_token_lambda": -0.3,
            "logit_scale": 50,
            "logit_weight": 1.0,
            "bg_idx": 0,
            "alpha": 0.02,
        }
    },
   
    "wbs_si": {
        "448": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.50,
            "logit_scale": 25,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        },
        "896": {
            "prob_thd": 0.10,
            "cls_token_lambda": -0.50,
            "logit_scale": 25,
            "logit_weight": 0.0,
            "bg_idx": 0,
            "alpha": 0.0,
        }
    },
    
}