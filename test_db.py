# test_db.py
from app.db import DatabaseManager


db = DatabaseManager("data/ner_kb.db")  
print("База данных инициализирована.\n")

# 1. Категории
db.add_category("PERSON", "Люди и персонажи" )
db.add_category("ORG", "Организации" )
db.add_category("ANIMAL", "Животные" )

cats = db.list_categories()
print(f"Количество категорий: {len(cats)}")
for c in cats:
    print(f"  {c['name']} — {c['description']}")

# 2. Сущности
db.add_entity(
    "Никола Тесла",
    "PERSON",
    description="Изобретатель и физик",
)

db.add_entity(
    "Tesla Inc.",
    "ORG",
)

db.add_entity(
    "Барсик",
    "ANIMAL",
)

ents = db.list_entities()
print(f"\nКоличество сущностей: {len(ents)}")
for e in ents:
    print(f"  [{e['category']}] {e['name']}")

# 3. Поиск
results = db.search_entity("Tesla")
print(f"\nРезультаты поиска по запросу 'Tesla': {len(results)}")
for r in results:
    print(f"  {r['name']} ({r['category']})")

# 4. Тексты и связи
t1 = db.add_text("Никола Тесла разработал переменный ток.", source="wikipedia")
t2 = db.add_text("Tesla Inc. выпустила новую модель электромобиля.")
t3 = db.add_text("Никола Тесла сотрудничал с Tesla Inc. в выдуманной истории.")

entity_person = db.get_entity("Никола Тесла", "PERSON")
entity_org = db.get_entity("Tesla Inc.", "ORG")

db.link_entity_to_text(entity_person["id"], t1, role="subject")
db.link_entity_to_text(entity_org["id"], t2, role="subject")
db.link_entity_to_text(entity_person["id"], t3, role="subject")
db.link_entity_to_text(entity_org["id"], t3, role="object")

print("\nТексты, связанные с сущностью 'Никола Тесла':")
for t in db.get_texts_for_entity("Никола Тесла"):
    print(f"  [{t['role']}] {t['content'][:60]}")

print(f"\nСущности, связанные с текстом #{t3}:")
for e in db.get_entities_in_text(t3):
    print(f"  [{e['role']}] {e['name']} ({e['category']})")

# 5. Обновление и смена категории

db.add_category("PET", "Домашние животные")
db.reassign_entity("Барсик", "PET")
reassigned = db.get_entity("Барсик")
print(f"Барсик переназначен в категорию: {reassigned['category']}")

# 6. Удаление
db.add_entity("Временная сущность", "PERSON")
db.delete_entity("Временная сущность")
assert db.get_entity("Временная сущность") is None
print("\nПроверка удаления сущности пройдена.")

db.add_category("TEMP_CAT", "Временная")
db.add_entity("Дочерняя сущность", "TEMP_CAT")
db.delete_category("TEMP_CAT")  # каскадно удалит дочернюю сущность
assert db.get_entity("Дочерняя сущность") is None
print("Проверка каскадного удаления категории пройдена.")

# 7. Итоговая статистика
print(f"\n{'=' * 40}")
print("Статистика:", db.stats())
print("Все проверки скрипта выполнены.")

db.close()
