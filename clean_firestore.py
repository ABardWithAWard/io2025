import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import sys

def clean_collections():
    try:
        # Initialize Firebase Admin SDK
        cred = credentials.Certificate('ocr/firebaseSecretKey.json')
        firebase_admin.initialize_app(cred)
        
        # Get Firestore client
        db = firestore.client()
        
        # Collections to clean
        collections = ['images', 'contacts']
        
        for collection_name in collections:
            print(f"Cleaning collection: {collection_name}")
            
            # Get all documents in the collection
            docs = db.collection(collection_name).stream()
            
            # Delete each document
            deleted_count = 0
            for doc in docs:
                doc.reference.delete()
                deleted_count += 1
            
            print(f"Deleted {deleted_count} documents from {collection_name}")
        
        print("Database cleaning completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Ask for confirmation before proceeding
    confirmation = input("This will delete ALL documents from 'images' and 'contacts' collections. Are you sure? (yes/no): ")
    
    if confirmation.lower() == 'yes':
        clean_collections()
    else:
        print("Operation cancelled.") 
