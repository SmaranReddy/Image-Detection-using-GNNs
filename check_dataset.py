from relation_prediction.vg_dataset import VGRelationshipDataset

dataset = VGRelationshipDataset(
    "./data/visual_genome/relationships.json",
    "./data/visual_genome/image_data.json",
)

print("\nFINAL SIZE:", len(dataset))
