CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets openearthmap -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 9 --name_path ./configs/cls_openearthmap.txt --text_shift --do_shift --per_label --dataset_name openearthmap --text_adjust True --save_result openearthmap.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets loveda -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 7 --name_path ./configs/cls_loveda.txt --text_shift --do_shift --per_label --dataset_name loveda --text_adjust True --save_result loveda.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets isaid -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 3e-3 --tta_steps 3 --num_classes 16 --name_path ./configs/cls_isaid.txt --text_shift --do_shift --per_label --dataset_name isaid --text_adjust True --save_result isaid.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets potsdam -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 2e-4 --tta_steps 3 --num_classes 6 --name_path ./configs/cls_potsdam.txt --text_shift --do_shift --per_label --dataset_name potsdam --text_adjust True --save_result potsdam.txt  

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets vaihingen -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 2e-2 --tta_steps 3 --num_classes 6 --name_path ./configs/cls_vaihingen.txt --text_shift --do_shift --per_label --dataset_name vaihingen --text_adjust True --save_result vaihingen.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets uavid -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 2e-4 --tta_steps 3 --num_classes 7 --name_path ./configs/cls_uavid.txt --text_shift --do_shift --per_label --dataset_name uavid --text_adjust True --save_result uavid.txt

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12347 shift_classification_remote.py <datapath> --test_sets udd5 -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 5 --name_path ./configs/cls_udd5.txt --text_shift --do_shift --per_label --dataset_name udd5 --text_adjust True --save_result udd5.txt

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12347 shift_classification_remote.py <datapath> --test_sets vdd -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 7 --name_path ./configs/cls_vdd.txt --text_shift --do_shift --per_label --dataset_name vdd --text_adjust True  --save_result vdd.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets whu_aerial -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 2 --name_path ./configs/cls_whu_aerial.txt --text_shift --do_shift --per_label --dataset_name whu_aerial --text_adjust True --save_result whu_aerial.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets whu_aerial -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 2 --resolution 896 --name_path ./configs/cls_whu_aerial.txt --text_shift --do_shift --per_label --dataset_name whu_aerial --text_adjust True --save_result whu_aerial_896.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets whu_sat -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 2 --name_path ./configs/cls_whu_sat.txt --text_shift --do_shift --per_label --dataset_name whu_sat --text_adjust True --save_result whu_sat.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets inria -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 2e-4 --tta_steps 3 --num_classes 2 --name_path ./configs/cls_inria.txt --text_shift --do_shift --per_label --dataset_name inria --text_adjust True --save_result inria.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets inria -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 2e-4 --tta_steps 3 --num_classes 2 --resolution 896 --name_path ./configs/cls_inria.txt --text_shift --do_shift --per_label --dataset_name inria --text_adjust True --save_result inria_896.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets xbd -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 7e-5 --tta_steps 3 --num_classes 2 --name_path ./configs/cls_xbd_prompt_v4_claude1-1.txt --text_shift --do_shift --per_label --dataset_name xbd --text_adjust True --save_result xbd.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets xbd -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 7e-5 --tta_steps 3 --num_classes 2 --resolution 896 --name_path ./configs/cls_xbd_prompt_v4_claude1-1.txt --text_shift --do_shift --per_label --dataset_name xbd --text_adjust True --save_result xbd_896.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets chn6-cug -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-5 --tta_steps 3 --num_classes 2 --name_path ./configs/cls_chn6-cug_prompt_v4_claude1-1.txt --text_shift --do_shift --per_label --dataset_name chn6-cug --text_adjust True --save_result chn6-cug.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets chn6-cug -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-5 --tta_steps 3 --num_classes 2 --resolution 896 --name_path ./configs/cls_chn6-cug_prompt_v4_claude1-1.txt --text_shift --do_shift --per_label --dataset_name chn6-cug --text_adjust True --save_result chn6-cug_896.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets deepglobe -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 5e-5 --tta_steps 3 --num_classes 2 --name_path ./configs/cls_deepglobe.txt --text_shift --do_shift --per_label --dataset_name deepglobe --text_adjust True --save_result deepglobe.txt 

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets deepglobe -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 5e-5 --tta_steps 3 --num_classes 2 --resolution 896 --name_path ./configs/cls_deepglobe.txt --text_shift --do_shift --per_label --dataset_name deepglobe --text_adjust True --save_result deepglobe_896.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets massachusetts -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 2 --name_path ./configs/cls_massachusetts.txt --text_shift --do_shift --per_label --dataset_name massachusetts --text_adjust True --save_result massachusetts.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets massachusetts -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 2 --resolution 896 --name_path ./configs/cls_massachusetts.txt --text_shift --do_shift --per_label --dataset_name massachusetts --text_adjust True --save_result massachusetts_896.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12346 shift_classification_remote.py <datapath> --test_sets spacenet -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 2 --name_path ./configs/cls_spacenet.txt --text_shift --do_shift --per_label --dataset_name spacenet --text_adjust True  --save_result spacenet.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12346 shift_classification_remote.py <datapath> --test_sets spacenet -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 2 --resolution 896 --name_path ./configs/cls_spacenet.txt --text_shift --do_shift --per_label --dataset_name spacenet --text_adjust True --save_result spacenet_896.txt   

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets wbs_si -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 2 --name_path ./configs/cls_wbs_si.txt --text_shift --do_shift --per_label --dataset_name wbs_si --text_adjust True --save_result wbs_si.txt   

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=12347 shift_classification_remote.py <datapath> --test_sets wbs_si -a ViT-B/16 -b 1 --seed 1 --img_aug --lr 1e-4 --tta_steps 3 --num_classes 2 --resolution 896 --name_path ./configs/cls_wbs_si.txt --text_shift --do_shift --per_label --dataset_name wbs_si --text_adjust True --save_result wbs_si_896.txt